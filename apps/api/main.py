"""
SmishShield API - hardened FastAPI wrapper around the ONNX smishing classifier.

Run (example):
    API_KEYS=key1,key2 uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --workers 2

Environment variables
---------------------
API_KEYS                    Comma-separated valid keys (required unless ALLOW_NO_AUTH=1)
ALLOW_NO_AUTH               "1" to disable auth (local development only)
MAX_CONCURRENT_INFERENCES   In-flight inference calls per worker       (default 4)
QUEUE_TIMEOUT_S             Seconds to wait for a free slot before 503 (default 2.0)
MAX_BODY_BYTES              Hard request-body cap in bytes             (default 2_000_000)
MODEL_DIR                   Override model directory
USE_INT8                    "0" to serve the fp32 model                (default 1)
INTRA_OP_THREADS            ORT threads per worker. With W workers on C cores,
                            use roughly C / W.
ENABLE_DOCS                 "0" to hide /docs and /redoc in production

Not handled here (do these at the gateway / reverse proxy):
TLS, per-client rate limiting, CORS policy, IP allow-listing.

Privacy: message bodies are never logged. Only counts, timings and request IDs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from starlette.datastructures import MutableHeaders

from ml.infer import InvalidInputError, SmishClassifier

logger = logging.getLogger("smishshield.api")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

MAX_MESSAGES_PER_REQUEST = 128
MAX_CHARS_PER_MESSAGE = 2000
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    api_keys: tuple[str, ...]
    allow_no_auth: bool
    max_concurrent: int
    queue_timeout_s: float
    max_body_bytes: int
    model_dir: str | None
    use_int8: bool
    intra_op_threads: int | None
    enable_docs: bool


def load_settings() -> Settings:
    keys = tuple(k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip())
    threads = os.getenv("INTRA_OP_THREADS")
    return Settings(
        api_keys=keys,
        allow_no_auth=os.getenv("ALLOW_NO_AUTH", "0") == "1",
        max_concurrent=int(os.getenv("MAX_CONCURRENT_INFERENCES", "4")),
        queue_timeout_s=float(os.getenv("QUEUE_TIMEOUT_S", "2.0")),
        max_body_bytes=int(os.getenv("MAX_BODY_BYTES", "2000000")),
        model_dir=os.getenv("MODEL_DIR") or None,
        use_int8=os.getenv("USE_INT8", "1") != "0",
        intra_op_threads=int(threads) if threads else None,
        enable_docs=os.getenv("ENABLE_DOCS", "1") != "0",
    )


SETTINGS = load_settings()


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
Message = Annotated[str, StringConstraints(max_length=MAX_CHARS_PER_MESSAGE)]


class ScanRequest(BaseModel):
    messages: list[Message] = Field(
        ..., min_length=1, max_length=MAX_MESSAGES_PER_REQUEST
    )
    echo_text: bool = Field(
        False, description="Include each original message in its result."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "messages": [
                        "Claim your Rs. 500 cashback at bit.ly/cash-xyz",
                        "Dinner at 8 pm?",
                    ],
                    "echo_text": False,
                }
            ]
        }
    )


class MessageResult(BaseModel):
    index: int
    text: str | None = None
    is_smishing: bool
    spam_probability: float
    threshold: float
    model_version: str
    variant: str


class ScanResponse(BaseModel):
    request_id: str
    count: int
    flagged_count: int
    results: list[MessageResult]


# --------------------------------------------------------------------------- #
# ASGI middleware (pure ASGI: no BaseHTTPMiddleware quirks)
# --------------------------------------------------------------------------- #
class _BodyTooLarge(Exception):
    pass


class BodySizeLimitMiddleware:
    """Rejects oversized bodies early, including chunked uploads."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def _reject(self, send) -> None:
        body = json.dumps({"detail": "Request body too large"}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            return await self._reject(send)

        received = 0
        response_started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        async def tracking_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if not response_started:
                await self._reject(send)


class RequestContextMiddleware:
    """Assigns a request ID and adds X-Request-ID / X-Process-Time-Ms headers."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = time.perf_counter()

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1e3:.1f}"
            await send(message)

        await self.app(scope, receive, send_with_headers)


# --------------------------------------------------------------------------- #
# App lifecycle
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not SETTINGS.api_keys and not SETTINGS.allow_no_auth:
        raise RuntimeError(
            "No API_KEYS configured. Set API_KEYS, or ALLOW_NO_AUTH=1 for local dev."
        )
    if SETTINGS.allow_no_auth:
        logger.warning("Authentication is DISABLED (ALLOW_NO_AUTH=1)")

    kwargs = {"use_int8": SETTINGS.use_int8, "intra_op_threads": SETTINGS.intra_op_threads}
    if SETTINGS.model_dir:
        kwargs["model_dir"] = SETTINGS.model_dir

    # Model load + warm-up happen once, off the event loop.
    app.state.classifier = await run_in_threadpool(lambda: SmishClassifier(**kwargs))
    app.state.inference_sem = asyncio.Semaphore(SETTINGS.max_concurrent)
    app.state.stats = {"requests": 0, "messages": 0, "flagged": 0, "rejected_busy": 0}
    logger.info("SmishShield API ready")
    yield
    app.state.classifier = None


app = FastAPI(
    title="SmishShield API",
    version="1.1.0",
    description="Multilingual smishing and fraud detection engine",
    lifespan=lifespan,
    docs_url="/docs" if SETTINGS.enable_docs else None,
    redoc_url="/redoc" if SETTINGS.enable_docs else None,
    openapi_url="/openapi.json" if SETTINGS.enable_docs else None,
)
# Added last = outermost, so 413 responses still get request-ID headers.
app.add_middleware(BodySizeLimitMiddleware, max_bytes=SETTINGS.max_body_bytes)
app.add_middleware(RequestContextMiddleware)


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


@app.exception_handler(InvalidInputError)
async def invalid_input_handler(request: Request, exc: InvalidInputError):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "request_id": _request_id(request)},
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    # Log the traceback server-side; never echo internals or message content.
    logger.exception("Unhandled error request_id=%s", _request_id(request))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": _request_id(request)},
    )


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    if SETTINGS.allow_no_auth and not SETTINGS.api_keys:
        return
    if key:
        provided = key.encode()
        # Check every key (no early exit) with constant-time comparison.
        matches = [secrets.compare_digest(provided, k.encode()) for k in SETTINGS.api_keys]
        if any(matches):
            return
    raise HTTPException(status_code=401, detail="Invalid or missing API key")


def get_classifier(request: Request) -> SmishClassifier:
    clf = getattr(request.app.state, "classifier", None)
    if clf is None:
        raise HTTPException(status_code=503, detail="Inference engine is not ready.")
    return clf


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health/live", tags=["ops"])
def live():
    """Liveness: the process is up."""
    return {"status": "ok"}


@app.get("/health/ready", tags=["ops"])
@app.get("/health", include_in_schema=False)  # backwards-compatible alias
def ready(request: Request):
    """Readiness: 200 only once the model is loaded and warmed up."""
    clf = getattr(request.app.state, "classifier", None)
    if clf is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "ready",
        "model_version": clf.model_version,
        "variant": clf.variant,
        "threshold": clf.threshold,
    }


@app.post(
    "/api/v1/scan",
    response_model=ScanResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(require_api_key)],
    tags=["scan"],
)
async def scan_messages(
    payload: ScanRequest,
    request: Request,
    clf: SmishClassifier = Depends(get_classifier),
):
    state = request.app.state
    request_id = _request_id(request)

    # Bound concurrent inference so bursts queue briefly, then shed load.
    try:
        await asyncio.wait_for(
            state.inference_sem.acquire(), timeout=SETTINGS.queue_timeout_s
        )
    except asyncio.TimeoutError:
        state.stats["rejected_busy"] += 1
        raise HTTPException(
            status_code=503,
            detail="Server busy, please retry shortly.",
            headers={"Retry-After": "1"},
        )

    started = time.perf_counter()
    try:
        predictions = await run_in_threadpool(clf.predict, payload.messages)
    finally:
        state.inference_sem.release()
    elapsed_ms = (time.perf_counter() - started) * 1e3

    results = [
        MessageResult(
            index=i,
            text=payload.messages[i] if payload.echo_text else None,
            is_smishing=pred["is_smishing"],
            spam_probability=pred["spam_probability"],
            threshold=pred["threshold"],
            model_version=pred["model_version"],
            variant=pred["variant"],
        )
        for i, pred in enumerate(predictions)
    ]
    flagged = sum(r.is_smishing for r in results)

    state.stats["requests"] += 1
    state.stats["messages"] += len(results)
    state.stats["flagged"] += flagged
    logger.info(
        "scan request_id=%s n=%d flagged=%d infer_ms=%.1f",
        request_id, len(results), flagged, elapsed_ms,
    )

    return ScanResponse(
        request_id=request_id,
        count=len(results),
        flagged_count=flagged,
        results=results,
    )


@app.get("/api/v1/stats", dependencies=[Depends(require_api_key)], tags=["ops"])
def stats(request: Request):
    """Per-worker counters. Watch the flagged ratio for drift or abuse."""
    s = request.app.state.stats
    ratio = s["flagged"] / s["messages"] if s["messages"] else 0.0
    return {**s, "flagged_ratio": round(ratio, 4)}
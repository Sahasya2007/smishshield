import os
import pytest
from fastapi.testclient import TestClient

os.environ["API_KEYS"] = "test-secret-key"
os.environ["ALLOW_NO_AUTH"] = "0"

from apps.api.main import SlidingWindowRateLimiter, app  # noqa: E402

AUTH_HEADER = {"X-API-Key": "test-secret-key"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoints(client: TestClient) -> None:
    live = client.get("/health/live")
    assert live.status_code == 200
    assert live.json()["status"] == "ok"

    ready = client.get("/health/ready")
    assert ready.status_code == 200
    body = ready.json()
    assert body["status"] == "ready"
    assert body["model_version"] == "indicbert-v1-int8"


def test_scan_requires_valid_api_key(client: TestClient) -> None:
    res_missing = client.post("/api/v1/scan", json={"messages": ["Hello"]})
    assert res_missing.status_code == 401

    res_wrong = client.post(
        "/api/v1/scan",
        headers={"X-API-Key": "wrong-key"},
        json={"messages": ["Hello"]},
    )
    assert res_wrong.status_code == 401


def test_scan_rejects_blank_strings_with_422(client: TestClient) -> None:
    res = client.post(
        "/api/v1/scan",
        headers=AUTH_HEADER,
        json={"messages": ["   "]},
    )
    assert res.status_code == 422


def test_scan_batch_classification(client: TestClient) -> None:
    payload = {
        "messages": [
            "URGENT: Your SBI YONO account will be blocked. Update KYC at http://bit.ly/sbi-kyc",
            "HDFC Bank: Rs. 500.00 debited from A/c **1234. UPI Ref 987654321012.",
        ],
        "echo_text": True,
    }
    res = client.post("/api/v1/scan", headers=AUTH_HEADER, json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["count"] == 2
    assert data["flagged_count"] == 1
    assert data["results"][0]["is_smishing"] is True
    assert data["results"][1]["is_smishing"] is False
    assert data["results"][1]["guard_applied"] == "dlt_transactional_guard"


def test_scan_rate_limiter_enforces_429(client: TestClient) -> None:
    original = client.app.state.rate_limiter
    client.app.state.rate_limiter = SlidingWindowRateLimiter(
        max_requests=2, window_seconds=60.0
    )
    try:
        for _ in range(2):
            r = client.post(
                "/api/v1/scan", headers=AUTH_HEADER, json={"messages": ["Hi"]}
            )
            assert r.status_code == 200

        blocked = client.post(
            "/api/v1/scan", headers=AUTH_HEADER, json={"messages": ["Hi"]}
        )
        assert blocked.status_code == 429
        assert "Retry-After" in blocked.headers
    finally:
        client.app.state.rate_limiter = original
import pytest
from ml.infer import InvalidInputError, SmishClassifier


@pytest.fixture(scope="module")
def classifier() -> SmishClassifier:
    return SmishClassifier(use_int8=True)


def test_empty_and_whitespace_short_circuit(classifier: SmishClassifier) -> None:
    results = classifier.predict(["", "   \t\n  "])
    assert len(results) == 2
    for res in results:
        assert res["is_smishing"] is False
        assert res["spam_probability"] == 0.0
        assert res["model_version"] == "indicbert-v1-int8"


def test_invalid_input_types_raise_error(classifier: SmishClassifier) -> None:
    with pytest.raises(InvalidInputError):
        classifier.predict(12345)  # type: ignore[arg-type]

    with pytest.raises(InvalidInputError):
        classifier.predict(["valid text", None])  # type: ignore[list-item]


def test_legitimate_indian_dlt_sms_not_flagged(classifier: SmishClassifier) -> None:
    legit_samples = [
        "HDFC Bank: Rs. 1,450.00 debited from A/c **4821 on 27-11-26 to VPA swiggy@axisbank. UPI Ref 632910482910.",
        "482910 is your OTP for SBI NetBanking login. Valid for 5 mins. Do not share with anyone.",
        "Dear Customer, your Jio plan of Rs 299 has been recharged successfully. Enjoy unlimited 5G data.",
        "IRCTC Ticket Booked: PNR 4521890321, Train 12801 PURUSHOTTAM EXP, 28-Nov, B2 42. Have a safe journey.",
        "Your Amazon order #408-9182736-1234561 has been delivered. Thank you for shopping with us.",
    ]
    results = classifier.predict(legit_samples)
    assert all(not r["is_smishing"] for r in results)


def test_smishing_and_homoglyphs_detected(classifier: SmishClassifier) -> None:
    lures = [
        "URGENT: Your SBI YONO account will be blocked today. Update PAN card immediately: http://bit.ly/sbi-kyc-update",
        "Aapka bijli connection aaj raat 9:30 baje kat diya jayega kyunki pichla bill update nahi hua. Call 9876543210",
        "Your bаnk аccоunt is suspended. Verify KYC at http://bit.ly/verify-nоw",
    ]
    results = classifier.predict(lures)
    assert all(r["is_smishing"] for r in results)
    assert all(r["spam_probability"] >= classifier.threshold for r in results)


def test_zero_width_obfuscation_benign(classifier: SmishClassifier) -> None:
    res = classifier.predict_one("H\u200be\u200bl\u200bl\u200bo, meeting at 5 pm?")
    assert res["is_smishing"] is False
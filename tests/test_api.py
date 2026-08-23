"""
CreditIQ API Tests
"""

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


# ============================================================
# HEALTH
# ============================================================

def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "CreditIQ API"

    assert "model_loaded" in data
    assert "database_connected" in data


# ============================================================
# SCORE
# ============================================================

def test_score_valid_applicant():

    response = client.post(
        "/score",
        json={
            "applicant_id": 447009
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["applicant_id"] == 447009

    assert (
        0.0
        <= data["probability_of_default"]
        <= 1.0
    )

    assert data["decision"] in {
        "APPROVE",
        "DECLINE",
    }

    assert (
        0.0
        <= data["threshold"]
        <= 1.0
    )

    assert (
        data["model_version"]
        == "xgboost_monotonic_v1"
    )


# ============================================================
# EXPLAIN
# ============================================================

def test_explain_valid_applicant():

    response = client.post(
        "/explain",
        json={
            "applicant_id": 447009
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["applicant_id"] == 447009

    assert (
        0.0
        <= data["probability_of_default"]
        <= 1.0
    )

    assert data["decision"] in {
        "APPROVE",
        "DECLINE",
    }

    assert isinstance(
        data["top_risk_factors"],
        list,
    )

    assert isinstance(
        data["top_protective_factors"],
        list,
    )


# ============================================================
# INVALID APPLICANT
# ============================================================

def test_score_unknown_applicant():

    response = client.post(
        "/score",
        json={
            "applicant_id": 999999999
        },
    )

    assert response.status_code == 404


# ============================================================
# REQUEST VALIDATION
# ============================================================

def test_score_rejects_invalid_applicant_id():

    response = client.post(
        "/score",
        json={
            "applicant_id": 0
        },
    )

    assert response.status_code == 422


def test_explain_rejects_invalid_applicant_id():

    response = client.post(
        "/explain",
        json={
            "applicant_id": -1
        },
    )

    assert response.status_code == 422
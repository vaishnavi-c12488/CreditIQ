"""
CreditIQ Health Route

Checks:
- API availability
- Model availability
- PostgreSQL connectivity
"""

from fastapi import APIRouter

from api.dependencies import (
    check_model_loaded,
    check_database_connection,
)

from api.schemas.response import HealthResponse


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get(
    "",
    response_model=HealthResponse,
)
def health_check():

    model_loaded = check_model_loaded()

    database_connected = check_database_connection()

    overall_status = (
        "healthy"
        if model_loaded and database_connected
        else "unhealthy"
    )

    return HealthResponse(
        status=overall_status,
        service="CreditIQ API",
        model_loaded=model_loaded,
        database_connected=database_connected,
    )
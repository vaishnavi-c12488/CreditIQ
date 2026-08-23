"""
CreditIQ FastAPI Application
"""

from fastapi import FastAPI

from api.routes.score import router as score_router
from api.routes.explain import router as explain_router
from api.routes.health import router as health_router

# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="CreditIQ API",
    description=(
        "Credit-risk scoring API using the finalized "
        "CreditIQ Monotonic XGBoost model."
    ),
    version="1.0.0",
)


# ============================================================
# ROUTES
# ============================================================

app.include_router(
    score_router
)
app.include_router(explain_router)
app.include_router(health_router)
# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "service": "CreditIQ API",
        "status": "running",
        "version": "1.0.0",
    }
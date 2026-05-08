

from fastapi import FastAPI
from app.services.health import get_health_status

app = FastAPI(title="Microservice API")

@app.get("/health")
def health_check():
    """Health check endpoint mapping to the services layer."""
    return get_health_status()
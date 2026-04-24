import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from dotenv import load_dotenv

from app.api import api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

app_logger = logging.getLogger("app")

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")


# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    app_logger.info("Starting up Kairos...")
    yield
    app_logger.info("Shutting down Kairos...")


# --- App Initialization ---
app = FastAPI(
    title="Kairos",
    description="A FastAPI service for Zoe",
    version="0.1.0",
    lifespan=lifespan,
)

# --- Routes ---
app.include_router(api.router)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

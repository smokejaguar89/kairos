import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


@router.get("/plant/{plant_id}/hoursuntildry")
async def get_hours_until_dry(plant_id: str):
    """Get the number of hours until a specific plant is dry."""
    return {"plant_id": plant_id, "hours_until_dry": 10}

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.predictor import CounterfeitPredictor
from app.schemas import PredictionResponse

# Works both locally (static/ next to app/) and in Docker (/app/static)
_HERE = Path(__file__).parent.parent  # counterfeit_service/
STATIC_DIR = str(os.getenv("STATIC_DIR", _HERE / "static"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

predictor = CounterfeitPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    yield


app = FastAPI(
    title="Counterfeit Detection Service",
    description="Multimodal counterfeit product detection for Ozon marketplace",
    version="1.0.0",
    lifespan=lifespan,
)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://sonyakrasovskaya.github.io",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(str(Path(STATIC_DIR) / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(..., description="Product photo"),
    name: str = Form("", description="Product name (name_rus)"),
    description: str = Form("", description="Product description"),
    brand: str = Form("", description="Brand name"),
    category: str = Form("", description="CommercialTypeName4 — product category string"),
    price: float = Form(0.0, description="PriceDiscounted"),
    item_time_alive: float = Form(0.0, description="Days on marketplace"),
    item_count_sales30: float = Form(0.0, description="Sales last 30 days"),
    item_count_returns30: float = Form(0.0, description="Returns last 30 days"),
    seller_time_alive: float = Form(0.0, description="Seller age in days"),
):
    # Validate image content type
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    # Tabular inputs — only the fields provided by the user.
    # All other feature_cols default to 0.0 in predictor._build_tabular_row.
    tab_inputs = {
        "CommercialTypeName4": category,
        "PriceDiscounted": price,
        "item_time_alive": item_time_alive,
        "item_count_sales30": item_count_sales30,
        "item_count_returns30": item_count_returns30,
        "seller_time_alive": seller_time_alive,
    }

    try:
        result = predictor.predict(
            image_bytes=image_bytes,
            name=name,
            description=description,
            brand=brand,
            tab_inputs=tab_inputs,
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return PredictionResponse(**result)

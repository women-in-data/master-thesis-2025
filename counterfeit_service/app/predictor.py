"""
Inference pipeline for Feature Fusion CatBoost counterfeit detection model.

Model expects columns in this exact order (confirmed from model.feature_names_):
  [0:38]   tabular features (feature_cols, CommercialTypeName4 is categorical string)
  [38:238] Doc2Vec 200-dim embeddings (d2v_0 .. d2v_199) — unscaled
  [238:750] CLIP image embeddings scaled (img_0 .. img_511)

Doc2Vec model (d2v_model.pkl) must be saved from the training notebook and placed
in the artifacts/ directory. If missing, d2v_ columns are filled with zeros and
a warning is logged — model still runs on tabular + image signal.
"""

import os
import logging
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from PIL import Image
import io

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent.parent  # counterfeit_service/
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", _HERE / "artifacts"))

# Fixed column counts verified from model.feature_names_
N_D2V = 200
N_IMG = 512
D2V_COLS = [f"d2v_{i}" for i in range(N_D2V)]
IMG_COLS = [f"img_{i}" for i in range(N_IMG)]

# Prediction threshold for is_counterfeit verdict
THRESHOLD = float(os.getenv("COUNTERFEIT_THRESHOLD", "0.5"))


class CounterfeitPredictor:
    def __init__(self):
        self.model: Optional[CatBoostClassifier] = None
        self.img_scaler = None
        self.feature_cols: list = []
        self.cat_cols: list = []
        self.d2v_model = None
        self._clip_model = None
        self._clip_processor = None
        self._clip_loaded = False

    def load(self):
        logger.info("Loading artifacts from %s", ARTIFACTS_DIR)

        # CatBoost model
        self.model = CatBoostClassifier()
        self.model.load_model(str(ARTIFACTS_DIR / "catboost_model.cbm"))
        logger.info("CatBoost loaded: %d features", len(self.model.feature_names_))

        # Tabular metadata
        self.feature_cols = joblib.load(ARTIFACTS_DIR / "feature_cols.pkl")
        self.cat_cols = joblib.load(ARTIFACTS_DIR / "cat_cols.pkl")
        logger.info("feature_cols: %d, cat_cols: %s", len(self.feature_cols), self.cat_cols)

        # Image scaler
        self.img_scaler = joblib.load(ARTIFACTS_DIR / "img_scaler.pkl")
        logger.info("img_scaler loaded, expects %d dims", self.img_scaler.n_features_in_)

        # Doc2Vec model (optional — saved from training notebook)
        d2v_path = ARTIFACTS_DIR / "d2v_model.pkl"
        if d2v_path.exists():
            self.d2v_model = joblib.load(d2v_path)
            logger.info("Doc2Vec model loaded")
        else:
            logger.warning(
                "d2v_model.pkl not found in artifacts/. "
                "Text modality (d2v_0..d2v_199) will be filled with zeros. "
                "Run save_d2v_model.py after training to save it."
            )

        # CLIP is loaded lazily on first image request
        logger.info("Predictor ready (CLIP will load on first request)")

    def _load_clip(self):
        if self._clip_loaded:
            return
        logger.info("Loading CLIP ViT-B/32 model (CPU)...")
        from transformers import CLIPModel, CLIPProcessor
        self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self._clip_model.eval()
        self._clip_loaded = True
        logger.info("CLIP loaded")

    def _get_image_embedding(self, image_bytes: bytes) -> np.ndarray:
        """PIL image bytes → CLIP 512-dim → img_scaler → array[512]"""
        import torch
        self._load_clip()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self._clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self._clip_model.get_image_features(**inputs)
        embedding = features[0].numpy().astype(np.float32)  # (512,)
        scaled = self.img_scaler.transform(embedding.reshape(1, -1))[0]
        return scaled  # (512,)

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """text → Doc2Vec infer_vector(200,) or zeros if model not available"""
        if self.d2v_model is not None:
            tokens = text.lower().split()
            if not tokens:
                return np.zeros(N_D2V, dtype=np.float32)
            vec = self.d2v_model.infer_vector(tokens, epochs=50)
            return np.array(vec, dtype=np.float32)
        return np.zeros(N_D2V, dtype=np.float32)

    def _build_tabular_row(self, tab_inputs: dict) -> pd.DataFrame:
        """
        Build a single-row DataFrame with all feature_cols.
        Missing numeric columns are filled with 0.0.
        CommercialTypeName4 stays as string.
        """
        row = {}
        for col in self.feature_cols:
            if col in self.cat_cols:
                row[col] = str(tab_inputs.get(col, ""))
            else:
                row[col] = float(tab_inputs.get(col, 0.0))
        return pd.DataFrame([row])[self.feature_cols]

    def _build_fused_df(
        self,
        tab_inputs: dict,
        d2v_vec: np.ndarray,
        img_vec: np.ndarray,
    ) -> pd.DataFrame:
        """Concatenate all modalities in training order: tabular → d2v → img"""
        df_tab = self._build_tabular_row(tab_inputs)
        df_d2v = pd.DataFrame([d2v_vec], columns=D2V_COLS)
        df_img = pd.DataFrame([img_vec], columns=IMG_COLS)
        fused = pd.concat([df_tab, df_d2v, df_img], axis=1)
        return fused

    def _predict_proba(self, fused_df: pd.DataFrame) -> float:
        prob = self.model.predict_proba(fused_df)[0][1]
        return float(prob)

    def predict(
        self,
        image_bytes: bytes,
        name: str,
        description: str,
        brand: str,
        tab_inputs: dict,
    ) -> dict:
        """
        Full inference pipeline.

        Returns:
            is_counterfeit, probability, signals (multimodal, image, text)
        """
        # Build combined text field same as training:
        # train_df['text'] = name_rus + ' ' + description + ' ' + brand_name
        text = f"{name} {description} {brand}".strip()

        # Get embeddings
        img_vec = self._get_image_embedding(image_bytes)   # (512,) scaled
        d2v_vec = self._get_text_embedding(text)           # (200,) unscaled

        # ── Main prediction (all modalities) ──
        fused = self._build_fused_df(tab_inputs, d2v_vec, img_vec)
        multimodal_score = self._predict_proba(fused)

        # ── Image-only signal: zero out d2v, keep img ──
        zeros_d2v = np.zeros(N_D2V, dtype=np.float32)
        tab_zero = {col: "" if col in self.cat_cols else 0.0 for col in self.feature_cols}
        tab_zero["CommercialTypeName4"] = str(tab_inputs.get("CommercialTypeName4", ""))
        fused_img_only = self._build_fused_df(tab_zero, zeros_d2v, img_vec)
        image_signal = self._predict_proba(fused_img_only)

        # ── Text-only signal: zero out img, keep d2v ──
        zeros_img = np.zeros(N_IMG, dtype=np.float32)
        fused_text_only = self._build_fused_df(tab_zero, d2v_vec, zeros_img)
        text_signal = self._predict_proba(fused_text_only)

        return {
            "is_counterfeit": multimodal_score >= THRESHOLD,
            "probability": round(multimodal_score, 4),
            "signals": {
                "multimodal_score": round(multimodal_score, 4),
                "image_signal": round(image_signal, 4),
                "text_signal": round(text_signal, 4),
            },
        }

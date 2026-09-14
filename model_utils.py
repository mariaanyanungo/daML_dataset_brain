import json
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "brain_mri_classifier.keras"
CONFIG_PATH = MODEL_DIR / "preprocessing_config.json"


UNCERTAINTY_THRESHOLD = 0.50


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def load_model():
    return keras.models.load_model(MODEL_PATH)


def preprocess_image(pil_image: Image.Image, config: dict) -> np.ndarray:
    img = pil_image.convert("RGB")
    h, w = config["img_size"]
    img = img.resize((w, h))  # PIL resize takes (width, height)
    arr = np.array(img).astype(np.float32)
    arr = arr * config["rescale"]
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict(model, config: dict, pil_image: Image.Image):
    x = preprocess_image(pil_image, config)
    probs = model.predict(x, verbose=0)[0]
    class_names = config["class_names"]

    top_idx = int(np.argmax(probs))
    predicted_class = class_names[top_idx]
    confidence = float(probs[top_idx])
    all_probs = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    is_uncertain = confidence < UNCERTAINTY_THRESHOLD

    return predicted_class, confidence, all_probs, is_uncertain
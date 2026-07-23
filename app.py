import gc
import json
import os
import threading
from pathlib import Path

# Reduce CPU thread and memory pressure before importing TensorFlow.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
STAGE1_MODEL_PATH = BASE_DIR / "stage1_model.keras"
STAGE2_MODEL_PATH = BASE_DIR / "stage2_model.keras"

print("Starting application...", flush=True)

with open(BASE_DIR / "settings.json", "r", encoding="utf-8") as f:
    settings = json.load(f)

IMG_SIZE = int(settings.get("IMG_SIZE", 380))
STAGE1_THRESHOLD = float(settings.get("STAGE1_THRESHOLD", 0.38))
CONFIDENCE_THRESHOLD = float(settings.get("CONFIDENCE_THRESHOLD", 0.75))

STAGE2_CLASS_NAMES = settings.get(
    "stage2_class_names",
    [
        "AMD",
        "Cataract",
        "Diabetic Retinopathy",
        "Glaucoma",
        "Hypertensive Retinopathy",
        "Pathological Myopia",
    ],
)

# Render's free service has limited RAM. The old version loaded both
# EfficientNetB4 models together, which could make prediction hang for a long
# time. A lock also prevents two users from loading models simultaneously.
PREDICTION_LOCK = threading.Lock()


def release_model(model):
    """Release a Keras model and ask Python/TensorFlow to free memory."""
    if model is not None:
        del model
    tf.keras.backend.clear_session()
    gc.collect()


def load_single_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file was not found: {model_path.name}")

    print(f"Loading {model_path.name}...", flush=True)
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"Loaded {model_path.name} successfully.", flush=True)
    return model


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def run_model(model_path: Path, image_batch: np.ndarray) -> np.ndarray:
    """Load one model, run inference, then release it before the next model."""
    model = None
    try:
        model = load_single_model(model_path)
        output = model(image_batch, training=False)
        return np.asarray(output)
    finally:
        release_model(model)


def predict_retinal_disease(image: Image.Image) -> str:
    if image is None:
        return "Please upload a retinal fundus image first."

    # Only one prediction at a time. This protects the small Render instance
    # from running out of memory when multiple visitors click Predict.
    with PREDICTION_LOCK:
        try:
            print("Prediction started.", flush=True)
            x = preprocess_image(image)

            # Load and release Stage 1 before Stage 2 is loaded.
            stage1_output = run_model(STAGE1_MODEL_PATH, x)
            diseased_prob = float(np.ravel(stage1_output)[0])
            normal_prob = 1.0 - diseased_prob

            stage1_prediction = (
                "Diseased"
                if diseased_prob >= STAGE1_THRESHOLD
                else "Normal"
            )

            lines = [
                "TWO-STAGE RETINAL DISEASE DETECTION RESULT",
                "=" * 55,
                "",
                "STAGE 1: Normal vs Diseased",
                "-" * 38,
                f"Prediction: {stage1_prediction}",
                f"Normal probability: {normal_prob:.4f}",
                f"Diseased probability: {diseased_prob:.4f}",
                f"Diseased threshold: {STAGE1_THRESHOLD:.2f}",
            ]

            if stage1_prediction == "Normal":
                lines.extend(
                    [
                        "",
                        "STAGE 2: Not Applied",
                        "-" * 38,
                        "Reason: Stage 1 predicted Normal.",
                    ]
                )
            else:
                # Stage 1 has already been removed from memory here.
                stage2_output = run_model(STAGE2_MODEL_PATH, x)[0]

                pred_id = int(np.argmax(stage2_output))
                confidence = float(np.max(stage2_output))

                if confidence < CONFIDENCE_THRESHOLD:
                    final_prediction = "Unknown/Uncertain"
                else:
                    final_prediction = STAGE2_CLASS_NAMES[pred_id]

                top_n = min(3, len(STAGE2_CLASS_NAMES))
                top_indices = np.argsort(stage2_output)[-top_n:][::-1]

                lines.extend(
                    [
                        "",
                        "STAGE 2: Disease Group Classification",
                        "-" * 45,
                        f"Prediction: {final_prediction}",
                        f"Confidence: {confidence:.4f}",
                        f"Confidence threshold: {CONFIDENCE_THRESHOLD:.2f}",
                        "",
                        f"Top {top_n} possible disease groups:",
                    ]
                )

                for rank, idx in enumerate(top_indices, start=1):
                    idx = int(idx)
                    probability = float(stage2_output[idx])
                    lines.append(
                        f"{rank}. {STAGE2_CLASS_NAMES[idx]}: "
                        f"{probability:.4f}"
                    )

            lines.extend(
                [
                    "",
                    "Important note:",
                    (
                        "This application is for research and screening "
                        "demonstration only. It is not a medical diagnosis."
                    ),
                ]
            )

            print("Prediction completed.", flush=True)
            return "\n".join(lines)

        except Exception as error:
            print(f"Prediction error: {type(error).__name__}: {error}", flush=True)
            tf.keras.backend.clear_session()
            gc.collect()
            return (
                "An error occurred while processing the image.\n\n"
                f"Error: {type(error).__name__}: {error}"
            )


with gr.Blocks(title="Two-Stage Retinal Disease Detection") as demo:
    gr.Markdown(
        """
# Two-Stage Retinal Disease Detection

Upload a retinal fundus image.

**Stage 1:** Normal vs Diseased  
**Stage 2:** Disease group classification

The first prediction after the service wakes up may take a few minutes because
Render must download and load the trained models.

This application is for research and screening demonstration only.
"""
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Fundus Image")
            predict_button = gr.Button("Predict", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(
                label="Prediction Output",
                lines=24,
                interactive=False,
            )

    predict_button.click(
        fn=predict_retinal_disease,
        inputs=input_image,
        outputs=output_text,
        concurrency_limit=1,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    print(f"Starting Gradio on port {port}...", flush=True)
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
    )

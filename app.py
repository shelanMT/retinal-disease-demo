import gc
import json
import os
import threading
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
STAGE1_MODEL_PATH = BASE_DIR / "stage1_model.tflite"
STAGE2_MODEL_PATH = BASE_DIR / "stage2_model.tflite"
SETTINGS_PATH = BASE_DIR / "settings.json"

print("Starting TensorFlow Lite application...", flush=True)

with open(SETTINGS_PATH, "r", encoding="utf-8") as file:
    settings = json.load(file)

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

PREDICTION_LOCK = threading.Lock()


def validate_files():
    missing = [
        path.name
        for path in (STAGE1_MODEL_PATH, STAGE2_MODEL_PATH, SETTINGS_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required files: " + ", ".join(missing)
        )


validate_files()
print("TensorFlow Lite model files found.", flush=True)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def prepare_input(array: np.ndarray, input_detail: dict) -> np.ndarray:
    dtype = input_detail["dtype"]

    if np.issubdtype(dtype, np.floating):
        return array.astype(dtype)

    scale, zero_point = input_detail.get("quantization", (0.0, 0))
    if scale and scale > 0:
        converted = np.round(array / scale + zero_point)
        limits = np.iinfo(dtype)
        converted = np.clip(converted, limits.min, limits.max)
        return converted.astype(dtype)

    return array.astype(dtype)


def prepare_output(array: np.ndarray, output_detail: dict) -> np.ndarray:
    dtype = output_detail["dtype"]

    if np.issubdtype(dtype, np.floating):
        return array.astype(np.float32)

    scale, zero_point = output_detail.get("quantization", (0.0, 0))
    if scale and scale > 0:
        return (array.astype(np.float32) - zero_point) * scale

    return array.astype(np.float32)


def run_tflite_model(model_path: Path, image_batch: np.ndarray) -> np.ndarray:
    interpreter = None

    try:
        print(f"Loading {model_path.name}...", flush=True)

        interpreter = tf.lite.Interpreter(
            model_path=str(model_path),
            num_threads=1,
        )
        interpreter.allocate_tensors()

        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]

        expected_shape = tuple(int(value) for value in input_detail["shape"])
        received_shape = tuple(image_batch.shape)

        if expected_shape != received_shape:
            raise ValueError(
                f"Input shape mismatch for {model_path.name}. "
                f"Expected {expected_shape}, received {received_shape}."
            )

        model_input = prepare_input(image_batch, input_detail)
        interpreter.set_tensor(input_detail["index"], model_input)
        interpreter.invoke()

        raw_output = interpreter.get_tensor(output_detail["index"])
        output = prepare_output(raw_output, output_detail)

        print(f"Finished {model_path.name} inference.", flush=True)
        return output

    finally:
        if interpreter is not None:
            del interpreter
        gc.collect()


def predict_retinal_disease(image: Image.Image) -> str:
    if image is None:
        return "Please upload a retinal fundus image first."

    with PREDICTION_LOCK:
        try:
            print("Prediction started.", flush=True)
            image_batch = preprocess_image(image)

            stage1_output = run_tflite_model(
                STAGE1_MODEL_PATH,
                image_batch,
            )

            diseased_prob = float(np.ravel(stage1_output)[0])
            diseased_prob = float(np.clip(diseased_prob, 0.0, 1.0))
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
                stage2_output = run_tflite_model(
                    STAGE2_MODEL_PATH,
                    image_batch,
                )[0]

                pred_id = int(np.argmax(stage2_output))
                confidence = float(np.max(stage2_output))

                final_prediction = (
                    "Unknown/Uncertain"
                    if confidence < CONFIDENCE_THRESHOLD
                    else STAGE2_CLASS_NAMES[pred_id]
                )

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
                    lines.append(
                        f"{rank}. {STAGE2_CLASS_NAMES[idx]}: "
                        f"{float(stage2_output[idx]):.4f}"
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

            print("Prediction completed successfully.", flush=True)
            return "\n".join(lines)

        except Exception as error:
            print(
                f"Prediction error: {type(error).__name__}: {error}",
                flush=True,
            )
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

This application uses optimized TensorFlow Lite models.

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

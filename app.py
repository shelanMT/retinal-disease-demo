import json
import os
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent

print("Starting application...", flush=True)

with open(BASE_DIR / "settings.json", "r", encoding="utf-8") as f:
    settings = json.load(f)

IMG_SIZE = int(settings.get("IMG_SIZE", 380))
STAGE1_THRESHOLD = float(settings.get("STAGE1_THRESHOLD", 0.38))
CONFIDENCE_THRESHOLD = float(
    settings.get("CONFIDENCE_THRESHOLD", 0.75)
)

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

# Models are not loaded during startup.
stage1_model = None
stage2_model = None


def load_models():
    global stage1_model, stage2_model

    if stage1_model is None:
        print("Loading Stage 1 model...", flush=True)
        stage1_model = tf.keras.models.load_model(
            BASE_DIR / "stage1_model.keras",
            compile=False,
        )
        print("Stage 1 model loaded successfully.", flush=True)

    if stage2_model is None:
        print("Loading Stage 2 model...", flush=True)
        stage2_model = tf.keras.models.load_model(
            BASE_DIR / "stage2_model.keras",
            compile=False,
        )
        print("Stage 2 model loaded successfully.", flush=True)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.asarray(image, dtype=np.float32)

    return np.expand_dims(image_array, axis=0)


def predict_retinal_disease(image: Image.Image) -> str:
    if image is None:
        return "Please upload a retinal fundus image first."

    try:
        # Models load only after Predict is clicked.
        load_models()

        x = preprocess_image(image)

        stage1_output = stage1_model.predict(x, verbose=0)
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
            stage2_output = stage2_model.predict(x, verbose=0)[0]

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

        return "\n".join(lines)

    except Exception as error:
        print(f"Prediction error: {error}", flush=True)

        return (
            "An error occurred while processing the image.\n\n"
            f"Error: {error}"
        )


with gr.Blocks(
    title="Two-Stage Retinal Disease Detection"
) as demo:

    gr.Markdown(
        """
# Two-Stage Retinal Disease Detection

Upload a retinal fundus image.

**Stage 1:** Normal vs Diseased  
**Stage 2:** Disease group classification

The first prediction may take longer because the models load
after you click **Predict**.

This application is for research and screening demonstration only.
"""
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                type="pil",
                label="Upload Fundus Image",
            )

            predict_button = gr.Button(
                "Predict",
                variant="primary",
            )

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
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))

    print(
        f"Starting Gradio immediately on port {port}...",
        flush=True,
    )

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True,
    )

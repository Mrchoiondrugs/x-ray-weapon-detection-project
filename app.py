"""
X-ray Weapon Detection App
--------------------------
Single-file Streamlit application that acts as BOTH frontend and backend:
 - Frontend: file upload, sliders, buttons, image/result rendering
 - Backend: loads a YOLO (.pt) model and runs inference in-process

Place your trained model file "weapondetect.pt" in the SAME folder as this
app.py before running / deploying.

Run locally:
    streamlit run app.py

Deploy:
    Push this folder (app.py, requirements.txt, packages.txt, weapondetect.pt)
    to a repo and deploy on Streamlit Community Cloud, or any host that can
    run `streamlit run app.py`.
"""

import io
import os
import time

import numpy as np
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="X-ray Weapon Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "weapondetect.pt"  # <-- your model file, same folder as app.py


# ---------------------------------------------------------------------------
# Backend: model loading (cached so it only loads once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


def run_inference(model, image: Image.Image, conf: float, iou: float):
    """Run YOLO inference on a PIL image and return the raw result object."""
    results = model.predict(
        source=np.array(image),
        conf=conf,
        iou=iou,
        verbose=False,
    )
    return results[0]


# ---------------------------------------------------------------------------
# Frontend: sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Detection Settings")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU threshold (NMS)", 0.0, 1.0, 0.45, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model file:** `{}`\n\n"
    "Make sure this file is placed in the same directory as `app.py`.".format(MODEL_PATH)
)

# ---------------------------------------------------------------------------
# Frontend: header
# ---------------------------------------------------------------------------
st.title("🛡️ X-ray Weapon Detection System")
st.caption(
    "Upload a baggage / body-scan X-ray image, then click **Predict** to run "
    "the detection model."
)

# ---------------------------------------------------------------------------
# Load model (with friendly error handling)
# ---------------------------------------------------------------------------
model = None
model_error = None
if not os.path.exists(MODEL_PATH):
    model_error = (
        f"Model file '{MODEL_PATH}' was not found in the app directory. "
        f"Upload/copy your trained weights there and reload the app."
    )
else:
    try:
        with st.spinner("Loading model..."):
            model = load_model(MODEL_PATH)
    except Exception as e:
        model_error = f"Failed to load model: {e}"

if model_error:
    st.error(model_error)
else:
    st.sidebar.success("✅ Model loaded")

# ---------------------------------------------------------------------------
# Frontend: image upload + predict button
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload X-ray image", type=["jpg", "jpeg", "png", "bmp", "tiff"]
)

col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

    with col1:
        st.subheader("Input")
        st.image(image, use_container_width=True)

    predict_clicked = st.button(
        "🔍 Predict", type="primary", use_container_width=True, disabled=model is None
    )

    if predict_clicked and model is not None:
        start = time.time()
        with st.spinner("Running detection..."):
            result = run_inference(model, image, conf_threshold, iou_threshold)
        elapsed = time.time() - start

        annotated_bgr = result.plot()          # numpy array, BGR, boxes drawn
        annotated_rgb = annotated_bgr[:, :, ::-1]

        with col2:
            st.subheader("Detection Result")
            st.image(annotated_rgb, use_container_width=True)

        boxes = result.boxes
        num_detections = 0 if boxes is None else len(boxes)

        st.markdown("---")
        if num_detections > 0:
            st.warning(f"⚠️ {num_detections} potential weapon(s) detected!")

            names = result.names
            rows = []
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].tolist()]
                rows.append(
                    {
                        "Class": names.get(cls_id, str(cls_id)),
                        "Confidence": f"{conf:.1%}",
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    }
                )
            st.subheader("Detections")
            st.table(rows)
        else:
            st.success("✅ No weapons detected.")

        st.caption(f"Inference time: {elapsed:.2f}s")
else:
    with col1:
        st.info("⬆️ Upload an image to get started.")

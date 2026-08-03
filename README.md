# 🛡️ X-ray Weapon Detection

An AI-powered web app that detects weapons in X-ray images (baggage / body-scan style) using a YOLO object detection model, built with Streamlit.

🔗 **Live Demo:** [https://x-ray-weapon-detection-project-gu8pf5klmkwrj9fxmzdtlb.streamlit.app/](https://x-ray-weapon-detection-project-gu8pf5klmkwrj9fxmzdtlb.streamlit.app/) <!-- TODO: replace with your actual Streamlit Cloud URL -->

---

## 📖 Overview

This project provides a simple, interactive interface for running weapon detection on X-ray scan images. Upload an image, click **Predict**, and the app returns an annotated image with bounding boxes, class labels, and confidence scores for any detected weapons.

The entire app — frontend (UI) and backend (model inference) — runs in a single `app.py` file powered by Streamlit and Ultralytics YOLO.

---

## ✨ Features

- 📤 Upload X-ray images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`)
- 🔍 One-click **Predict** button to run inference
- 🎚️ Adjustable confidence and IoU (NMS) thresholds via sidebar
- 🖼️ Side-by-side view of input image and annotated detection result
- 📊 Detailed results table (class, confidence, bounding box coordinates)
- ⚡ Cached model loading for fast repeated predictions
- ☁️ Ready to deploy on Streamlit Community Cloud

---

## 🖥️ Screenshot

<!-- TODO: add a screenshot or GIF of the app in action -->
<!-- ![App screenshot](assets/demo.png) -->

---

## 🛠️ Tech Stack

| Component     | Technology                     |
|---------------|---------------------------------|
| Frontend/UI   | Streamlit                       |
| Backend/Inference | Ultralytics YOLO (PyTorch)  |
| Image handling| Pillow, NumPy, OpenCV           |
| Deployment    | Streamlit Community Cloud       |

---

## 📁 Project Structure

```
weapon-detection/
├── app.py              # Main Streamlit app (frontend + backend)
├── requirements.txt    # Python dependencies
├── packages.txt        # System-level dependencies (for cloud deployment)
├── weapondetect.pt      # Trained YOLO model weights (add this yourself)
└── README.md
```

> ⚠️ **Note:** `weapondetect.pt` is not included in this repo (model weights are typically too large / private). You must add your own trained model file to the project root before running or deploying.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/weapon-detection.git
cd weapon-detection
```

### 2. Add your model

Place your trained `weapondetect.pt` file in the project root (same folder as `app.py`).

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this repository to GitHub, including `weapondetect.pt` (or host it externally and load it at startup if it's too large for GitHub).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select this repository, branch, and set the main file path to `app.py`.
4. Deploy. Streamlit Cloud will automatically install packages from `requirements.txt` and `packages.txt`.
5. Once live, copy your app's URL and update the **Live Demo** link at the top of this README.

---

## ⚙️ Configuration

Detection thresholds can be adjusted directly from the app's sidebar:

- **Confidence threshold** – minimum confidence score for a detection to be shown
- **IoU threshold** – controls non-maximum suppression (overlap filtering) for bounding boxes

---

## ⚠️ Disclaimer

This tool is intended for **research, security screening, and educational purposes only** (e.g. airport/baggage scanning assistance). It is not a certified safety system and should not be used as the sole basis for security decisions. Detection accuracy depends entirely on the quality and scope of the underlying trained model.

---

## 📄 License

<!-- TODO: choose a license, e.g. MIT -->
This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io)

# 🚦 Real-Time Traffic Surveillance & Vehicle Analytics System

An intelligent, real-time computer vision application for automated traffic monitoring, interactive Region of Interest (ROI) definition, multi-class vehicle detection, persistent tracking, and bi-directional vehicle counting powered by **YOLOv8** and **OpenCV**.

---

## 📌 Overview

This project provides an end-to-end traffic surveillance pipeline designed to process video feeds from fixed traffic or surveillance cameras. Users can dynamically draw custom polygonal Regions of Interest (ROI) directly on the camera feed. The system tracks vehicles as they traverse through the defined zone, classifies them into distinct categories, counts exited vehicles without duplicates, and exports the annotated output stream.

---

## ✨ Features

- **🎯 Interactive Polygon ROI Selection**:
  - Dynamically draw any custom polygonal monitoring zone on the live video frame using left-clicks, then press `ENTER` to lock the boundary.
- **⚡ YOLOv8 Object Tracking**:
  - Leverages state-of-the-art `yolov8m` with persistent multi-object tracking (`persist=True`) to maintain consistent vehicle IDs across frames.
- **🚘 Multi-Category Classification**:
  - **Passenger Vehicles**: Cars and personal automobiles.
  - **Heavy Vehicles**: Buses and trucks filtered by bounding box area thresholds.
  - **Two-Wheelers**: Motorcycles and bicycles filtered for size accuracy.
- **🔢 Accurate Exit Counting**:
  - Point-in-polygon tests determine entry into the ROI. Tracks unique vehicle IDs and registers exits to prevent duplicate counts.
- **📊 Real-Time HUD Overlay**:
  - Live counts inside the ROI.
  - Cumulative total of passed vehicles per vehicle class.
  - Real-time processing speed monitor (FPS).
- **💾 Automated Video Export**:
  - Automatically records the annotated detection video with timestamped filenames in the `output/` folder.
- **⚙️ Hardware Acceleration**:
  - Automatically detects and utilizes CUDA GPU if available, with seamless fallback to CPU.

---

## 🗂️ Project Structure

```text
├── input/
│   └── .gitkeep             # Place source video files here (e.g. tf_video.mp4)
├── output/
│   └── .gitkeep             # Automatically generated detection videos saved here
├── main.py                  # Core surveillance, tracking & counting pipeline
├── requirements.txt         # Python package dependencies
├── .gitignore               # Excludes large media (>100MB) & model weights
└── README.md                # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/aditya-121105/traffic-surveillance-system.git
cd traffic-surveillance-system
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Default Input Video
Place your input video file inside the `input/` folder named `tf_video.mp4`:
```bash
python main.py
```

### Custom Video Path
You can also supply any custom video file path via command-line argument:
```bash
python main.py path/to/your_traffic_video.mp4
```

---

## 🎮 Interactive Controls

| Action | Control |
|---|---|
| **Add ROI Point** | `Left Click` on the video window |
| **Complete ROI Polygon** | Press `ENTER` key |
| **Exit / Terminate Program** | Press `ESC` key |

---

## 📦 Model Information & Weights

The application automatically downloads the official Ultralytics YOLOv8 medium model (`yolov8m.pt`) on its initial run. Model weights are cached locally and are excluded from git tracking via `.gitignore`.

---

## 👤 Author

- **Aditya Kumarkhaniya** - [@aditya-121105](https://github.com/aditya-121105)

# RAINFO — Flash Flood Early Warning System (FFEWS)

An intelligent, physics-informed AI Flash Flood Early Warning System engineered for high-altitude mountainous terrains (Himalayan regions, Northeast India, and Western Ghats). Built for disaster management authorities (NDMA/SDMA) and citizens.

---

## 📌 Key Capabilities

- **Hybrid Physics-Informed ML Engine**: Combines calibrated hydrological equations (SCS-CN runoff, Antecedent Precipitation Index, Kirpich time-of-concentration) with a soft-voting ensemble (LightGBM + GradientBoosting + LogisticRegression members, chosen by CV from a 5-model comparison that also includes RandomForest and XGBoost).
- **Reproducible End-to-End ML Pipeline (`ml/pipeline/`)**: one command runs data validation → physics feature engineering → EDA (5 charts + summary) → 5-model training → CV + held-out test evaluation → ensemble export. Fixed seed, versioned environment, artifacts written for serving.
- **Full Metric Reporting**: accuracy, balanced accuracy, precision, recall, F1, F2, ROC-AUC, PR-AUC, MCC, Cohen's kappa, log loss, Brier score, confusion matrix, best-F1 operating threshold, 5-fold CV mean±std, and ROC/PR curves — surfaced in the Model Lab UI and `ml_models/metrics.json`.
- **Model Lab UI (`/model`)**: model comparison table, ROC/PR curve charts, feature importances, EDA gallery, and a live prediction playground with sliders and historical scenario presets.
- **NDMA/IMD 4-Tier Hazard Alert Scale**: Classifies risk into `LOW` (Green), `MODERATE` (Yellow), `HIGH` (Orange), and `SEVERE` (Red Emergency).
- **NDMA Officer Console (`/officer`)**: Operational dashboard with live Leaflet topography, flood probability gauges, and emergency alert broadcast controls.
- **Citizen Public Portal (`/user`)**: Mobile-friendly public portal with district risk heatmaps, flood safety guidelines, and emergency helpline directories.
- **Offline SMS Relay (`/android_relay`)**: Field-deployable, zero-dependency offline SMS relay that broadcasts SMS alerts via local phone hotspot even if cellular internet towers collapse.
- **Mobile Alert PWA (`/mobile`)**: Progressive Web App with Firebase Cloud Messaging (FCM) push alerts and siren beeps.
- **Interactive Simulations**: Replays historical disaster timelines (e.g. Kedarnath 2013) and simulates live IoT river/rain gauge telemetry.

---

## 📂 Repository Structure

```text
├── backend/                  # FastAPI Core Engine & Services
│   ├── app/
│   │   ├── api/              # REST Endpoints (alerts, predict, rainfall, sensors)
│   │   ├── models/           # Hydrological physics, feature engineering & predictor
│   │   ├── services/         # SMS (Fast2SMS), FCM Push, POC Directory, Offline Relay
│   │   ├── utils/            # Logging and schema validations
│   │   ├── config.py         # App configurations
│   │   └── main.py           # FastAPI entrypoint & static route mounting
│   ├── requirements.txt      # Python backend dependencies
│   └── .env.example          # Environment variable template
├── mobile/                   # Progressive Web App (PWA) for mobile alerts
│   ├── index.html            # Citizen alert receiver UI
│   ├── fcm-sw.js             # Service Worker for push notifications & alarm audio
│   ├── manifest.json         # PWA Manifest
│   └── icons/                # PWA App icons
├── android_relay/            # Offline SMS Android Termux Server
│   ├── phone_server.py       # Standalone HTTP server for offline SMS forwarding
│   └── README.md             # Android hotspot & Termux setup instructions
├── ml/                       # Machine Learning Training Pipeline
│   ├── pipeline/             # Reproducible end-to-end pipeline package
│   │   ├── config.py         # Paths, seed, feature contract, model registry
│   │   ├── data.py           # Dataset load/merge/validation + physics features
│   │   ├── eda.py            # EDA charts + machine-readable summary
│   │   ├── train.py          # 5 models + soft-voting ensemble, full metric suite
│   │   └── run.py            # Orchestrator: python -m ml.pipeline.run
│   ├── scripts/              # Thin entrypoints (train_model.py, eda.py)
│   ├── tests/test_pipeline.py# Assert-based pipeline self-check
│   ├── reports/              # EDA charts, training report, eda_summary.json
│   ├── data/                 # Raw and engineered feature datasets
│   └── model_report.json     # Full metrics: comparison, CV, curves, importances
├── ml_models/                # Exported serving artifacts (flood_model.pkl, scaler.pkl, metrics.json)
├── model.html                # Model Lab UI: comparison, curves, EDA, prediction playground
├── demo/                     # Demonstration & Simulation Scripts
│   ├── kedarnath_demo.py     # Step-by-step Kedarnath 2013 disaster scenario replay
│   └── sensor_simulator.py   # IoT sensor stream simulator
├── docs/                     # Specifications & Research Documentation
│   ├── Architecture.md       # Full System Architecture
│   ├── Design.md             # Detailed Technical Design
│   └── PRD.md                # Product Requirements Document
├── portal.html               # Gateway Landing Page (Citizen / Officer selection)
├── index.html                # NDMA Officer Operational Console
├── user.html                 # Citizen Early Warning Portal
├── districts.html            # Monitored Districts & Evacuation Shelter Directory
├── dataset.csv               # Calibrated hydrological dataset
├── India_Flood_Incidents_Data.csv # Historical flood benchmarks
├── test_system.py            # Physics & model assertion self-check test suite
├── test_push.py              # Push notification assertion test suite
├── start_system.bat          # Windows one-click launcher
└── .gitignore                # Clean GitHub ignore rules
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+ installed
- Git

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies (backend + ML pipeline)
pip install -r backend/requirements.txt
pip install -r ml/requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `backend/.env` and update required keys (optional for local testing):
```bash
cp backend/.env.example backend/.env
```

### 4. Run Verification Self-Checks
Verify hydrological calculations, ML predictors, and push notification services:
```bash
python test_system.py
python test_push.py
```

### 5. Launch Application
**Windows One-Click:**
Double click `start_system.bat` or run:
```bash
start_system.bat
```

**Manual Launch:**
```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🌐 Web Endpoints

Once running, access the portals in your browser:

| Portal | URL | Description |
| :--- | :--- | :--- |
| **Gateway** | `http://localhost:8000/` | Entry portal to select Citizen or Officer mode |
| **Model Lab** | `http://localhost:8000/model` | Model comparison, metrics, EDA, live prediction playground |
| **Public Citizen Portal** | `http://localhost:8000/user` | Real-time threat tiers, safety guidelines, and maps |
| **NDMA Officer Console** | `http://localhost:8000/officer` | Command center for monitoring and broadcasting alerts |
| **Districts Directory** | `http://localhost:8000/districts`| Monitored districts & shelter directory |
| **Mobile Alert PWA** | `http://localhost:8000/mobile` | Citizen alert subscriber app |
| **Interactive API Docs**| `http://localhost:8000/docs` | OpenAPI / Swagger interactive documentation |

---

## 🧪 Demos & Testing

### Run the Reproducible ML Pipeline (EDA + Training + Artifacts)
```bash
python -m ml.pipeline.run              # full pipeline (EDA charts + training)
python -m ml.pipeline.run --no-eda     # training only (re-runs)
python ml/scripts/eda.py               # EDA-only
python ml/tests/test_pipeline.py       # pipeline self-check (assert-based)
```
Outputs: `ml/reports/` (EDA charts + `eda_summary.json` + `training_report.json`),
`ml_models/flood_model.pkl` + `scaler.pkl` + `metrics.json` (serving artifacts,
auto-copied into `backend/app/models/`), and `ml/model_report.json`.

### Physics & Model Self-Checks
```bash
python test_system.py
python test_push.py
```

### Replay Kedarnath 2013 Flash Flood Event
Simulate the timeline of the Kedarnath disaster to verify early warning trigger thresholds:
```bash
python demo/kedarnath_demo.py
```

### Simulate Real-time IoT Sensors
Stream live rainfall and soil moisture readings into the API:
```bash
python demo/sensor_simulator.py
```

---

## 📄 License
This project was developed for disaster risk reduction and early warning applications.

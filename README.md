# upskillCampus — Data Science & Machine Learning Internship

**Industrial Internship Report — Multi-Domain Predictive Modelling Projects**<br>
Prepared by **Shrrivathsan S**<br>
Internship Program: upSkill Campus (USC) & The IoT Academy, in association with UniConverge Technologies Pvt Ltd (UCT)<br>
Domain: Data Science and Machine Learning

---

## 📄 Final Report

The full internship report — including problem statements, design, source code, performance results, learnings, and future scope — is available here:<br>
`Internship_Report_Shrrivathsan_S.pdf`

Individual weekly progress reports are also included:<br>
`Shrrivathsan_Week-01.pdf`<br>
`Shrrivathsan_Week-02.pdf`<br>
`Shrrivathsan_Week-03.pdf`<br>
`Shrrivathsan_Week-04.pdf`

---

## 🚀 Projects

| Week | Project | Domain | Technique |
|------|---------|--------|-----------|
| 1 | Crop Production Prediction | Agriculture | Regression (Random Forest / Linear Regression) |
| 2 | Turbofan Engine RUL Prediction | Predictive Maintenance | Regression on NASA C-MAPSS time-series data |
| 3 | Smart City Traffic Forecasting | Smart City | Gradient Boosting time-series forecasting |
| 4 | Crop and Weed Detection | Precision Farming | Computer Vision (YOLOv8 object detection) |

### Week 1 — Agriculture Crop Production Prediction

Predicts crop production in India (2001–2014) from crop, variety, state, season, and cultivation cost, using a Random Forest Regressor with a Linear Regression baseline.<br>
**Metrics:** MAE, R² Score<br>
**Run:** `python crop_production_prediction.py --data datafile.csv`

### Week 2 — Turbofan Engine Remaining Useful Life (RUL) Prediction

Estimates remaining operational cycles before failure using NASA's C-MAPSS dataset (FD001–FD004), with piece-wise linear RUL labelling, rolling-window sensor features, and a per-engine health-status scoring system (Healthy / Monitor / Warning / Critical).<br>
**Best model:** Random Forest (outperformed Linear Regression across all four sub-datasets)<br>
**Run:** `python rul_prediction.py --train train_FD001.txt --test test_FD001.txt --rul RUL_FD001.txt`

### Week 3 — Smart City Traffic Forecasting

Forecasts hourly vehicle counts across four city junctions using a Gradient Boosting Machine trained on calendar, cyclical (sin/cos), and peak-hour/weekend features.<br>
**Metric:** Cross-validated MAE (vehicles/hour)<br>
**Run:** `python traffic_forecasting.py --data train.csv --target Vehicles`

### Week 4 — Crop and Weed Detection (YOLOv8)

Builds the full data pipeline for a YOLOv8-based crop/weed object detector: image resizing (512×512), bbox-aware augmentation via Albumentations (546 → 1,300 images), and train/val/test splitting for YOLO training.<br>
**Run:** `python crop_weed_pipeline.py`<br>
**Preprocessing/splitting only:** `python crop_weed_pipeline.py --skip-training`

---

## 🛠️ Setup

```bash
pip install -r shrrivathsan_ml_projects/requirements.txt
```

> **Note:** Dataset files are not included in this repository. See each project's README for the expected data path and format.

---

## 📁 Repository Structure

```
upskillCampus/
├── README.md
├── Internship_Report_Shrrivathsan_S.pdf
├── Shrrivathsan_Week-01.pdf
├── Shrrivathsan_Week-02.pdf
├── Shrrivathsan_Week-03.pdf
├── Shrrivathsan_Week-04.pdf
└── shrrivathsan_ml_projects/
    ├── README.md
    ├── requirements.txt
    ├── week01_crop_production/
    │   ├── README.md
    │   └── crop_production_prediction.py
    ├── week02_turbofan_rul/
    │   ├── README.md
    │   └── rul_prediction.py
    ├── week03_traffic_forecasting/
    │   ├── README.md
    │   └── traffic_forecasting.py
    └── week04_crop_weed_detection/
        ├── README.md
        └── crop_weed_pipeline.py
```

---

## 🙏 Acknowledgements

Thanks to the mentors and program coordinators at upSkill Campus, The IoT Academy, and UniConverge Technologies Pvt Ltd (UCT) for their guidance and structured curriculum throughout this internship.

# H&M Retail Intelligence Pipeline

End-to-end retail analytics on 1.37M customers and 106K SKUs using the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) Kaggle dataset.

---

## What This Project Does

Takes raw H&M transaction, customer, and product data and builds a full retail intelligence pipeline — from data cleaning to a statistically validated campaign business case.

---

## Pipeline Overview

| Phase | What it does |
|-------|-------------|
| 1 — Data Engineering | Chunk-loads 31M transactions, filters to 100K customers, saves as Parquet |
| 2 — Feature Engineering | Builds RFM features, price bands, seasonal tags, category preferences per customer |
| 3 — NLP Style Clustering | TF-IDF + KMeans on product descriptions → 12 style groups |
| 4 — RFM Segmentation | KMeans on RFM → 4 cohorts: Champions, Potential Loyalists, At Risk, Lost |
| 5 — Churn Classifier | XGBoost churn model (AUC 0.8477) with SHAP explainability |
| 6 — Recommender System | ALS collaborative filtering (MAP@12 0.0043) with cold-start fallback |
| 7 — Demand Forecasting | Prophet with multiplicative seasonality on top 4 product categories |
| 8 — A/B Test Simulation | Two-proportion z-test on At Risk reactivation campaign with revenue estimate |

---

## Key Results

- **Churn Model ROC-AUC:** 0.8477 (leakage-free — recency and RFM scores excluded)
- **Recommender MAP@12:** 0.0043 (baseline ALS on sparse implicit feedback)
- **A/B Test:** Statistically significant lift at 95% confidence (p < 0.05)
- **Segmentation:** 4 RFM cohorts × 12 NLP style groups = compound targeting labels

---

## Tech Stack

```
pandas · numpy · scikit-learn · xgboost · shap
implicit · prophet · scipy · sentence-transformers
```

---

## How to Run

All notebooks run on **Kaggle (free tier)** — no local setup needed.

1. Go to [Kaggle](https://www.kaggle.com) and open a new notebook
2. Add the H&M dataset as a data source
3. Run notebooks in order: Phase 1 → Phase 2 → ... → Phase 8
4. Each notebook reads Parquet files saved by the previous one

---

## Project Structure

```
├── Phase_1_Data_Engineering.ipynb
├── Phase_2_Feature_Engineering.ipynb
├── Phase_3_NLP_Style_Clustering.ipynb
├── Phase_4_RFM_Segmentation.ipynb
├── Phase_5_Churn_Classifier.ipynb
├── Phase_6_Recommender_System.ipynb
├── Phase_7_Demand_Forecasting.ipynb
├── Phase_8_AB_Test_Simulation.ipynb
└── README.md
```

---

## Notable Design Decisions

**Data leakage catch:** Initial churn model hit AUC 0.99 because R_score is a monotonic transform of recency — the exact variable used to define churn. Removed recency and all RFM scores. Honest AUC: 0.8477.

**Hybrid recommender:** ALS handles existing customers. New/cold-start customers fall back to top-selling items from their NLP style cluster — reusing Phase 3 output.

**Multiplicative seasonality:** Fashion demand scales with trend level. Additive seasonality systematically underestimates peak inventory needs.

**Historically grounded A/B baseline:** Repurchase rate derived from actual midpoint observation window, not assumed.

---

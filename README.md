# 🌧️ Rain-Tomorrow Prediction – Australia Weather

Accurately predicting whether it will rain tomorrow can help farmers, commuters, and emergency services make better decisions.
This repository contains the full **data-mining pipeline** I built for Assignment 3 of *31250 Introduction to Data Analytics* at UTS. It moves from raw Bureau-of-Meteorology data through to a **Soft-Voting ensemble** that achieved a public **Kaggle score of 0 · 81126**.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [End-to-End Approach](#end-to-end-approach)
4. [Why Each Choice?](#why-each-choice)
5. [Experimental Results](#experimental-results)
6. [Future Work](#future-work)

---

## Problem Statement

Predict the binary target **`RainTomorrow`** (Yes / No) for the next day using \~10 years of daily weather observations collected across Australia (51 199 rows, 22 columns).

---

## Dataset

* **16 numerical** + **5 categorical** predictors (wind directions, etc.)
* Pronounced **class imbalance** (≈22 % “Rain”).
* Up to **47 % missingness** in several sensor-based columns (e.g. *Sunshine*).

---

## End-to-End Approach

| Stage                        | Key Techniques                                                                            |
| ---------------------------- | ----------------------------------------------------------------------------------------- |
| **Exploration**              | Descriptive stats, correlation heat-map.                                                  |
| **Split**                    | 90 / 10 train–test *before* any transforms to avoid leakage.                              |
| **Missing-Value Imputation** | *MICE* (IterativeImputer) for numerics; **mode** for categoricals.                        |
| **Outliers**                 | 1.5 × IQR detection ➜ **Winsorize** at 5ᵗʰ & 95ᵗʰ percentiles.                            |
| **Encoding**                 | **Label Encoding** (after benchmarking vs One-Hot & Binary).                              |
| **Class Imbalance**          | **Bootstrapped up-sampling** of the minority class *only on train*.                       |
| **Scaling**                  | **RobustScaler** for outlier-heavy cols; **StandardScaler** for the rest.                 |
| **Feature Selection**        | **RFE** with Random Forest → top 15 features.                                             |
| **Modeling**                 | Baseline grid of 6 algorithms → tune **RF, XGBoost, MLP** with **HalvingRandomSearchCV**. |
| **Ensembling**               | Soft-Voting (weights 4 : 1 : 2 for XGB : MLP : RF).                                       |
| **Evaluation**               | Stratified 5-fold CV on F1; full test-set metrics; Kaggle submission.                     |

---

## Why Each Choice?

* **MICE > Mean/Median/KNN** – preserves multivariate relationships ➜ +F1.
* **Bootstrapped up-sampling** beat SMOTE / SMOTE-NC by giving cleaner decision boundaries while preventing synthetic noise.
* **HalvingRandomSearchCV** explored wider hyper-parameter space with 75 % less computation than exhaustive GridSearch.
* **Soft Voting** captured XGBoost’s precision on class 0, MLP’s recall on class 1, and RF’s overall balance.

---

## Experimental Results

| Model                     | Class 1 Precision | Class 1 Recall | Class 1 F1  | ROC-AUC     |
| ------------------------- | ----------------- | -------------- | ----------- | ----------- |
| **Random Forest (tuned)** | **0 · 71**        | 0 · 56         | 0 · 625     | 0 · 875     |
| **XGBoost (tuned)**       | 0 · 56            | 0 · 76         | 0 · 642     | **0 · 881** |
| **MLP (tuned)**           | 0 · 50            | **0 · 79**     | 0 · 611     | 0 · 868     |
| **Soft-Voting Ensemble**  | 0 · 63            | 0 · 70         | **0 · 721** | 0 · 883     |

*Public leaderboard*: **0 · 81126** F1.
*Confusion-matrix & ROC plots* can be found in the notebook.

Future Work
Cost-sensitive loss or focal loss to push minority-class F1 higher.

Try temporal models (Temporal Fusion Transformer, LSTM) to exploit seasonality.

Deploy as a REST API (FastAPI + Docker) and automate a daily forecast pipeline.


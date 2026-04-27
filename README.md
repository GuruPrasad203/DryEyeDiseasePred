# Dry Eye Disease Prediction

A machine learning project to predict Dry Eye Disease from patient health and lifestyle data using a Random Forest classifier.
Dry Eye Disease Dataset, uploaded by Daksh Nagra, Kaggle.
Available at: https://www.kaggle.com/datasets/dakshnagra/dry-eye-disease
---

## Overview

This notebook builds a binary classification model to predict whether a patient has Dry Eye Disease based on clinical measurements, lifestyle habits, sleep patterns, and eye symptom data. The best model achieves **77.89% accuracy** on the balanced dataset.

---

## Dataset

**File:** `Dry_Eye_Dataset.csv`  
**Rows:** 20,000 samples  
**Target:** `Dry Eye Disease` (binary: Yes/No)

### Features (26 columns)

| Category | Features |
|---|---|
| Demographics | Gender, Age, Height, Weight |
| Sleep | Sleep duration, Sleep quality, Sleep disorder, Wake up during night, Feel sleepy during day |
| Lifestyle | Stress level, Caffeine consumption, Alcohol consumption, Smoking, Physical activity, Daily steps |
| Clinical | Blood pressure, Heart rate, Medical issue, Ongoing medication |
| Digital habits | Average screen time, Smart device before bed, Blue-light filter |
| Eye symptoms | Discomfort Eye-strain, Redness in eye, Itchiness/Irritation in eye |

---

## Project Structure

```
DryEyeDisease/
├── dryeyedis.ipynb                          # Main notebook
├── Dry_Eye_Dataset.csv                      # Raw dataset
├── Dry_Eye_Dataset_normalized.csv           # Normalized version
├── Dry_Eye_Dataset_normalized_balanced.csv  # Balanced + normalized
├── Dry_Eye_Dataset_features.csv             # Feature-engineered dataset
├── best_rf_model_features.pkl               # Saved RF model (original)
├── best_rf_model_balanced_features.pkl      # Saved RF model (balanced)
├── feature_importance_rf.csv                # Feature importances (original)
├── feature_importance_rf_balanced.csv       # Feature importances (balanced)
├── scaled_features_distribution.png        # Distribution plots
├── scaled_features_boxplot.png             # Box plots
└── scaled_features_correlation.png         # Correlation heatmap
```

---

## Pipeline

### 1. Data Loading & Exploration
- Load normalized CSV, check shape, missing values, and class distribution

### 2. Feature Engineering
10 categories of engineered features are created on top of the raw columns:

- **Blood Pressure:** `BP_difference`, `BP_mean`, `BP_ratio`
- **Sleep Quality:** `Sleep_efficiency`, `Sleep_deficit`, `Sleep_interruption_score`
- **Eye Symptoms:** `Eye_symptom_sum`, `Eye_symptom_mean`, `Eye_symptom_max`, `Eye_symptom_severity`
- **Screen Time:** `Screen_exposure`, `Digital_strain`, `Unprotected_screen_time`
- **Lifestyle Risk:** `Lifestyle_risk_sum`, `Lifestyle_risk_mean`, `Lifestyle_risk_score`
- **Physical Activity:** `Activity_level`, `Sedentary_score`, `Cardiac_stress`
- **Sleep Disorders:** `Sleep_problem_score`, `Sleep_disturbance`
- **Medical:** `Medical_complexity`, `Health_risk`
- **Age-related:** `Age_risk`, `Age_health_interaction`
- **Composite:** `Dry_eye_risk_score` (mean of top risk factors)

### 3. Model Training
- **Train/test split:** 80/20 with stratification
- **Hyperparameter tuning:** `RandomizedSearchCV` (50 iterations, 5-fold CV)
- **Tuned parameters:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`, `class_weight`

### 4. Evaluation
- Accuracy, AUC-ROC, classification report, confusion matrix
- 5-fold cross-validation on the best model
- Feature importance analysis (top 20 features)

### 5. Ensemble Methods Compared

| Method | Accuracy |
|---|---|
| **Random Forest (Balanced) — Best** | **77.89%** |
| RF + Extended Tuning | 76.82% |
| Voting Ensemble | 76.78% |
| RF + Feature Selection | 76.55% |
| RF + Polynomial Features | 76.41% |
| Stacking Classifier | 76.41% |
| RF + Outlier Removal | 75.92% |

---

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
```

Install with:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

---

## Usage

1. Place `Dry_Eye_Dataset_normalized.csv` and `Dry_Eye_Dataset_normalized_balanced.csv` in the working directory
2. Open `dryeyedis.ipynb` in Jupyter
3. Run all cells in order

To load the saved model for inference:
```python
import joblib
model = joblib.load('best_rf_model_balanced_features.pkl')
predictions = model.predict(X_new)
```

---

## Results

The best Random Forest model trained on the balanced, feature-engineered dataset achieves:

- **Test Accuracy:** 77.89%
- **AUC-ROC:** reported per run (see notebook output)
- **Cross-validation mean accuracy** reported with std deviation

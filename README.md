# Regressor: Assessing Student Performance in Secondary Education

**Author:** Marwan Abdalfadeel  
**Course:** Python in Machine Learning and Data Science  
**Supervisor:** dr hab. inż. Ziemowit Dworakowski  
**Institution:** AGH University of Science and Technology  

---

## Project Overview

This project investigates factors influencing student academic performance using the **Student Performance dataset** from the UCI Machine Learning Repository.

The main research question is:

> Does increased study time weaken the negative effect of absences on the final grade (G3)?

To answer this question, two regression approaches were implemented:

- Linear Regression (with an explicit interaction term)
- Multilayer Perceptron (MLP) Regressor

Both Mathematics and Portuguese datasets were analyzed separately.

---

## Dataset

Source: UCI Machine Learning Repository  
Dataset ID: 320 – Student Performance  

Files used:
- `data/student-mat.csv`
- `data/student-por.csv`

Each dataset contains 33 variables including:

- Demographic features
- Behavioral features (e.g., studytime, absences, alcohol consumption)
- Academic features (G1, G2, G3)

Target variable:
- **G3** (Final grade, scale 0–20)

---

## Methodology

### Feature Selection

The following features were used for modeling:

- `absences`
- `studytime`
- `absences × studytime` (interaction term)
- `G2` (control variable)

The interaction term was constructed to explicitly test the moderation hypothesis.

---

### Linear Regression

- Model: `sklearn.linear_model.LinearRegression`
- Data split: 60% train / 20% validation / 20% test
- Evaluation metrics:
  - RMSE
  - R²
- Residual vs predicted plots generated using `PredictionErrorDisplay`
- Counterfactual experiment conducted by:
  - Fixing absences at 20
  - Fixing G2 at median value
  - Varying studytime from 1 to 4

---

### MLP Regressor

- Model: `sklearn.neural_network.MLPRegressor`
- Train/Test split: 80% / 20%
- 5-fold cross-validation
- Hyperparameter tuning using `GridSearchCV`
- Optimized metric: RMSE
- Same counterfactual procedure as Linear Regression

---

## Repository Structure

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, PredictionErrorDisplay

def run_regression_for_file(csv_path, dataset_name):

    print("\n" + "=" * 50)
    print(f"DATASET: {dataset_name}")
    print("=" * 50)

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    df = pd.read_csv(csv_path, sep=";")
    df.columns = df.columns.str.strip()

    # Column names
    abs_col = "absences"
    study_col = "studytime"
    g2_col = "G2"
    g3_col = "G3"

    # ------------------------------------------------------------
    # Interaction term (hypothesis)
    # ------------------------------------------------------------
    df["abs_x_study"] = df[abs_col] * df[study_col]

    # Features + target
    features = [abs_col, study_col, "abs_x_study", g2_col]
    X = df[features].values
    y = df[g3_col].values

    # ------------------------------------------------------------
    # Train / Validation / Test split (60 / 20 / 20)
    # ------------------------------------------------------------
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=1
    )

    # ------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # ------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------
    y_pred_train = reg.predict(X_train)
    y_pred_val = reg.predict(X_val)
    y_pred_test = reg.predict(X_test)

    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------
    display = PredictionErrorDisplay.from_estimator(
        reg,
        X_test,
        y_test,
        kind="residual_vs_predicted"
    )
    plt.title(f"Prediction Error Display ({dataset_name})")
    plt.show()

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)
    r2_test = r2_score(y_test, y_pred_test)

    interaction_coef = reg.coef_[features.index("abs_x_study")]

    print(f"RMSE (train/val/test): {rmse_train:.4f} / {rmse_val:.4f} / {rmse_test:.4f}")
    print(f"R^2  (train/val/test): {r2_train:.4f} / {r2_val:.4f} / {r2_test:.4f}")
    print(f"Interaction coef (absences × studytime): {interaction_coef:.6f}")

    if interaction_coef > 0:
        print("Hypothesis: SUPPORTED (studytime compensates absences)")
    else:
        print("Hypothesis: NOT supported")

    # ------------------------------------------------------------
    # COUNTERFACTUAL EXPERIMENT
    # ------------------------------------------------------------
    print("\nCounterfactual experiment (Linear Regression)")

    g2_baseline = float(np.median(df[g2_col].values))

    absences_fixed = 20
    studytime_values = [1, 2, 3, 4]

    rows = []
    for s in studytime_values:
        rows.append({
            "absences": absences_fixed,
            "studytime": s,
            "abs_x_study": absences_fixed * s,
            "G2": g2_baseline
        })

    cf_df = pd.DataFrame(rows)
    X_cf = cf_df[features].values

    pred_cf = reg.predict(X_cf)
    cf_df["predicted_G3"] = pred_cf

    print(cf_df[["absences", "studytime", "predicted_G3"]])
    print("=" * 50)


# ------------------------------------------------------------
# Run BOTH datasets
# ------------------------------------------------------------
# Ensure these files exist in your local path or environment
run_regression_for_file("student-mat.csv", "Math")
run_regression_for_file("student-por.csv", "Portuguese")

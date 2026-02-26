import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def mlp_regressor_lab6(csv_path, dataset_name):

    print("\n" + "=" * 60)
    print(f"DATASET: {dataset_name}")
    print("=" * 60)

    # ------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------
    df = pd.read_csv(csv_path, sep=";")
    df.columns = df.columns.str.strip()

    # ------------------------------------------------------------
    # Features (same as Linear model)
    # ------------------------------------------------------------
    df["abs_x_study"] = df["absences"] * df["studytime"]

    features = ["absences", "studytime", "abs_x_study", "G2"]
    X = df[features].values
    y = df["G3"].values

    # ------------------------------------------------------------
    # Train / Test split (test held out)
    # ------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    # ------------------------------------------------------------
    # Cross-validation (Lab 6 style metrics)
    # ------------------------------------------------------------
    reg0 = MLPRegressor(random_state=1, max_iter=2000)

    scores = cross_validate(
        reg0,
        X_train,
        y_train,
        cv=5,
        scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
    )

    cv_r2 = scores["test_r2"].mean()
    cv_mae = (-scores["test_neg_mean_absolute_error"]).mean()
    cv_rmse = (-scores["test_neg_root_mean_squared_error"]).mean()

    # ------------------------------------------------------------
    # Grid Search (Task 6.6)
    # ------------------------------------------------------------
    parameters = {
        "hidden_layer_sizes": [(20,), (50,), (50, 20)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01]
    }

    grid = GridSearchCV(
        reg0,
        parameters,
        cv=5,
        scoring="neg_root_mean_squared_error"
    )

    grid.fit(X_train, y_train)

    # IMPORTANT: final trained model
    best_reg = grid.best_estimator_

    # ------------------------------------------------------------
    # Final evaluation on test set
    # ------------------------------------------------------------
    y_pred = best_reg.predict(X_test)

    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_test = r2_score(y_test, y_pred)
    mae_test = mean_absolute_error(y_test, y_pred)

    # ------------------------------------------------------------
    # COUNTERFACTUAL EXPERIMENT (FOR COMPARISON WITH LINEAR)
    # ------------------------------------------------------------
    g2_baseline = float(np.median(df["G2"].values))

    absences_fixed = 20
    studytime_values =  [1, 2, 3, 4]
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

    pred_cf_mlp = best_reg.predict(X_cf)
    cf_df["predicted_G3_MLP"] = pred_cf_mlp

    # ------------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------------
    print("\nMLP REGRESSION RESULTS")
    print("-" * 60)
    print(f"CV (5-fold) mean:   R^2={cv_r2:.4f} | MAE={cv_mae:.4f} | RMSE={cv_rmse:.4f}")
    print(f"Best parameters:   {grid.best_params_}")
    print(f"TEST metrics:      R^2={r2_test:.4f} | MAE={mae_test:.4f} | RMSE={rmse_test:.4f}")

    print("\nCounterfactual experiment (MLP – comparison):")
    print(cf_df[["absences", "studytime", "predicted_G3_MLP"]])

    print("=" * 60)


# ------------------------------------------------------------
# Run BOTH datasets
# ------------------------------------------------------------
mlp_regressor_lab6("/content/student-mat.csv", "Math")
mlp_regressor_lab6("/content/student-por.csv", "Portuguese")

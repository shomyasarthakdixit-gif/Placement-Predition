import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def print_section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def run_advanced_regression():
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(root_path)
    data_path = os.path.join(project_root, "Data", "placement_predict_50k Dataset (2).csv")
    
    REGRESSION_FOLDER = os.path.join(project_root, "Output", "models")
    os.makedirs(REGRESSION_FOLDER, exist_ok=True)
    
    print_section("13. ADVANCED REGRESSION MODEL - SALARY PACKAGE")
    
    df = pd.read_csv(data_path)
    REGRESSION_TARGET = "Salary Package"

    regression_df = df[
        df[REGRESSION_TARGET].notna()
        &
        (pd.to_numeric(df[REGRESSION_TARGET], errors="coerce") > 0)
    ].copy()

    regression_df[REGRESSION_TARGET] = pd.to_numeric(regression_df[REGRESSION_TARGET], errors="coerce")
    regression_df = regression_df.dropna(subset=[REGRESSION_TARGET])

    regression_drop = ["StudentID", "PlacementStatus", "Salary Package", "IsAnomaly", "CGPA_Tier"]
    regression_drop = [column for column in regression_drop if column in regression_df.columns]

    X_regression = regression_df.drop(columns=regression_drop)
    y_regression = regression_df[REGRESSION_TARGET]

    print_section("14. ADVANCED REGRESSION - TRAIN TEST SPLIT")

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_regression, y_regression, test_size=0.20, random_state=42
    )

    reg_numeric_features = X_regression.select_dtypes(include=np.number).columns.tolist()
    reg_categorical_features = X_regression.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    reg_numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    reg_categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    reg_preprocessor = ColumnTransformer(transformers=[
        ("numeric", reg_numeric_pipeline, reg_numeric_features),
        ("categorical", reg_categorical_pipeline, reg_categorical_features)
    ], remainder="drop")

    X_reg_train_processed = reg_preprocessor.fit_transform(X_reg_train)
    X_reg_test_processed = reg_preprocessor.transform(X_reg_test)
    reg_feature_names = reg_preprocessor.get_feature_names_out()

    joblib.dump(reg_preprocessor, os.path.join(REGRESSION_FOLDER, "advanced_regression_preprocessor.pkl"))

    print_section("15. MODEL 1 - RANDOM FOREST REGRESSOR")

    # Using 100 estimators and limited depth to avoid extreme overfitting, 
    # but still highly accurate.
    advanced_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    advanced_model.fit(X_reg_train_processed, y_reg_train)

    y_pred_adv = advanced_model.predict(X_reg_test_processed)

    adv_mse = mean_squared_error(y_reg_test, y_pred_adv)
    adv_rmse = np.sqrt(adv_mse)
    adv_mae = mean_absolute_error(y_reg_test, y_pred_adv)
    adv_r2 = r2_score(y_reg_test, y_pred_adv)

    print(f"MSE  : {adv_mse:.4f}")
    print(f"RMSE : {adv_rmse:.4f}")
    print(f"MAE  : {adv_mae:.4f}")
    print(f"R2   : {adv_r2:.4f}")

    joblib.dump(advanced_model, os.path.join(REGRESSION_FOLDER, "advanced_regression.pkl"))

    # Save feature importances
    importance_table = pd.DataFrame({
        "Feature": reg_feature_names,
        "Importance": advanced_model.feature_importances_
    })
    
    importance_table["Importance_Percent"] = importance_table["Importance"] * 100
    importance_table = importance_table.sort_values(by="Importance", ascending=False)

    importance_table.to_csv(os.path.join(REGRESSION_FOLDER, "feature_importances.csv"), index=False)
    
    print("\n[SUCCESS] Advanced Regression artifacts saved successfully!")

if __name__ == "__main__":
    run_advanced_regression()

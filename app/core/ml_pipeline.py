import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import category_encoders as ce

def load_data_and_train(root_path):
    print("[ML] Starting Advanced Pipeline Training...")
    
    # 1. Load Data
    data_path = os.path.join(os.path.dirname(root_path), "Data", "placement_predict_50k Dataset (2).csv")
    df = pd.read_csv(data_path)
    
    # 2. Define Columns
    _numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    _categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    _TARGET_CLASS = 'PlacementStatus'
    _TARGET_REG = 'Salary Package'
    _DROP_COLS = ['StudentID', 'IsAnomaly', 'CGPA_Tier', _TARGET_CLASS, _TARGET_REG]
    
    feat_df = df.drop(columns=_DROP_COLS, errors='ignore').copy()
    _feature_cols = feat_df.columns.tolist()
    
    # Split data for classification
    X = feat_df
    y_cls = df[_TARGET_CLASS]
    
    # 3. Define Advanced Categorical Encoders
    # Ordinal: CollegeTier
    ordinal_cols = ['CollegeTier']
    # OHE: Gender, Hostel, HistoryOfBacklogs (Low Cardinality)
    ohe_cols = ['Gender', 'Hostel', 'HistoryOfBacklogs']
    # Target Encoding: City, Specialisation (High Cardinality)
    target_cols = ['City', 'Specialisation']
    # Hashing (Pseudo-Embedding): Stream
    hash_cols = ['Stream']
    
    # Define Numeric scalers
    clean_num_cols = ['CGPA', 'AttendancePercent']
    robust_num_cols = [c for c in _feature_cols if c not in (_categorical_cols + clean_num_cols)]
    
    # 4. Build Pipelines for each column type
    clean_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    robust_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    # We apply a basic 'most_frequent' imputer before encoding categorical data
    ord_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('enc', ce.OrdinalEncoder()) # Let it infer or pass mapping if needed
    ])
    
    ohe_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('enc', ce.OneHotEncoder(handle_unknown='value'))
    ])
    
    target_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('enc', ce.TargetEncoder(min_samples_leaf=20, smoothing=10))
    ])
    
    hash_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('enc', ce.HashingEncoder(n_components=8)) # Hash into 8 vector dimensions
    ])
    
    # 5. Assemble the Master Preprocessor
    # Filter the cols to ensure they exist in _feature_cols
    def filter_existing(cols):
        return [c for c in cols if c in _feature_cols]

    preprocessor = ColumnTransformer(transformers=[
        ('num_clean', clean_pipe, filter_existing(clean_num_cols)),
        ('num_robust', robust_pipe, filter_existing(robust_num_cols)),
        ('cat_ord', ord_pipe, filter_existing(ordinal_cols)),
        ('cat_ohe', ohe_pipe, filter_existing(ohe_cols)),
        ('cat_tgt', target_pipe, filter_existing(target_cols)),
        ('cat_hsh', hash_pipe, filter_existing(hash_cols))
    ], remainder='drop') # Drop anything we missed explicitly

    # 6. Train Models
    rf_clf_model = Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    print("[ML] Training Classification Model (RF)...")
    rf_clf_model.fit(X, y_cls)
    
    lr_clf_model = Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    print("[ML] Training Classification Model (LR)...")
    lr_clf_model.fit(X, y_cls)
    
    # 7. Regression Model (Trained only on placed students)
    placed_mask = df[_TARGET_CLASS] == 1
    X_reg = feat_df.loc[placed_mask]
    y_reg = df.loc[placed_mask, _TARGET_REG]
    
    # We can't use TargetEncoder cleanly for a different target without rebuilding the transformer.
    # To keep it simple, we'll train the regressor using the same preprocessor (fitted on classification target). 
    # Warning: TargetEncoder inside preprocessor is fitted on y_cls. Since we fit the pipeline again here,
    # if we pass y_reg, it will re-fit the TargetEncoder on y_reg, which is perfect!
    rf_reg_model = Pipeline([
        ('prep', preprocessor),
        ('reg', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    print("[ML] Training Regression Model (RF)...")
    rf_reg_model.fit(X_reg, y_reg)
    
    print("[ML] Training Complete!")
    
    return {
        'df': df,
        'rf_clf': rf_clf_model,
        'lr_clf': lr_clf_model,
        'rf_reg': rf_reg_model,
        'feature_cols': _feature_cols,
        'numeric_cols': _numeric_cols,
        'categorical_cols': _categorical_cols,
        'preprocessor': preprocessor
    }

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import category_encoders as ce
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
    
    # Define Multinomial Target
    salary_median = df.loc[df[_TARGET_REG] > 0, _TARGET_REG].median()
    def make_package_tier(salary):
        if salary < 3.0:
            return "Not Placed"
        elif salary < salary_median:
            return "Standard Package"
        return "Premium Package"
        
    y_multi = df[_TARGET_REG].apply(make_package_tier)
    
    # 3. Define Advanced Categorical Encoders
    ordinal_cols = ['CollegeTier']
    ohe_cols = ['Gender', 'Hostel', 'HistoryOfBacklogs']
    target_cols = ['City', 'Specialisation']
    hash_cols = ['Stream']
    
    # Define Numeric scalers
    minmax_cols = ['AptitudeTestScore', 'CodingTestScore']
    clean_num_cols = ['CGPA', 'AttendancePercent']
    robust_num_cols = [c for c in _feature_cols if c not in (_categorical_cols + clean_num_cols + minmax_cols)]
    
    # 4. Build Pipelines for each column type
    clean_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean', add_indicator=True)),
        ('scaler', StandardScaler())
    ])
    
    robust_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    minmax_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])
    
    ord_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('enc', ce.OrdinalEncoder())
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
        ('enc', ce.HashingEncoder(n_components=8))
    ])
    
    # 5. Assemble the Master Preprocessor
    def filter_existing(cols):
        return [c for c in cols if c in _feature_cols]

    preprocessor = ColumnTransformer(transformers=[
        ('num_clean', clean_pipe, filter_existing(clean_num_cols)),
        ('num_robust', robust_pipe, filter_existing(robust_num_cols)),
        ('num_minmax', minmax_pipe, filter_existing(minmax_cols)),
        ('cat_ord', ord_pipe, filter_existing(ordinal_cols)),
        ('cat_ohe', ohe_pipe, filter_existing(ohe_cols)),
        ('cat_tgt', target_pipe, filter_existing(target_cols)),
        ('cat_hsh', hash_pipe, filter_existing(hash_cols))
    ], remainder='drop')

    # 6. Train Models
    print("[ML] Training Classification Models...")
    
    rf_clf = Pipeline([('prep', preprocessor), ('clf', RandomForestClassifier(n_estimators=30, max_depth=12, random_state=42, n_jobs=-1))])
    rf_clf.fit(X, y_cls)
    
    lr_clf = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))])
    lr_clf.fit(X, y_cls)
    
    ridge_clf = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(penalty='l2', max_iter=1000, random_state=42))])
    ridge_clf.fit(X, y_cls)

    lasso_clf = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=42))])
    lasso_clf.fit(X, y_cls)

    elastic_clf = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000, random_state=42))])
    elastic_clf.fit(X, y_cls)

    dt_clf = Pipeline([('prep', preprocessor), ('clf', DecisionTreeClassifier(max_depth=10, random_state=42))])
    dt_clf.fit(X, y_cls)

    gb_clf = Pipeline([('prep', preprocessor), ('clf', GradientBoostingClassifier(n_estimators=50, random_state=42))])
    gb_clf.fit(X, y_cls)

    xgb_clf = Pipeline([('prep', preprocessor), ('clf', XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss', random_state=42))])
    xgb_clf.fit(X, y_cls)
    
    # Multinomial
    softmax_clf = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42))])
    softmax_clf.fit(X, y_multi)
    
    # 7. Regression Model
    placed_mask = df[_TARGET_CLASS] == 1
    X_reg = feat_df.loc[placed_mask]
    y_reg = df.loc[placed_mask, _TARGET_REG]
    
    rf_reg = Pipeline([('prep', preprocessor), ('reg', RandomForestRegressor(n_estimators=30, max_depth=12, random_state=42, n_jobs=-1))])
    rf_reg.fit(X_reg, y_reg)
    
    # 8. Unsupervised Models
    print("[ML] Training Unsupervised Models (M4)...")
    # Use the preprocessor fitted inside the rf_clf pipeline
    fitted_prep = rf_clf.named_steps['prep']
    X_trans = fitted_prep.transform(X)
    
    pca = PCA(n_components=2, random_state=42)
    pca_comps = pca.fit_transform(X_trans)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_trans)

    print("[ML] Training Complete!")
    
    return {
        'df': df,
        'feature_cols': _feature_cols,
        'numeric_cols': _numeric_cols,
        'categorical_cols': _categorical_cols,
        'preprocessor': preprocessor,
        'rf_clf': rf_clf,
        'lr_clf': lr_clf,
        'ridge_clf': ridge_clf,
        'lasso_clf': lasso_clf,
        'elastic_clf': elastic_clf,
        'dt_clf': dt_clf,
        'gb_clf': gb_clf,
        'xgb_clf': xgb_clf,
        'softmax_clf': softmax_clf,
        'rf_reg': rf_reg,
        'kmeans': kmeans,
        'pca': pca,
        'X_trans': X_trans,
        'pca_comps': pca_comps
    }

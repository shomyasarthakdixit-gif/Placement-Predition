import numpy as np
from flask import current_app

def run_prediction_for_model(form, model_type):
    """Build feature row from form, run chosen model, return result dict."""
    ml_data = current_app.config['ML_PIPELINE']
    _feature_cols = ml_data['feature_cols']
    
    # Models mapping based on model_type
    models = {
        'rf': ml_data.get('rf_clf'),
        'lr': ml_data.get('lr_clf'),
        'ridge': ml_data.get('ridge_clf'),
        'lasso': ml_data.get('lasso_clf'),
        'elasticnet': ml_data.get('elastic_clf'),
        'dt': ml_data.get('dt_clf'),
        'gb': ml_data.get('gb_clf'),
        'xgb': ml_data.get('xgb_clf'),
        'softmax': ml_data.get('softmax_clf')
    }
    rf_reg_model = ml_data.get('rf_reg')
    
    clf_model = models.get(model_type)
    if not clf_model:
        return {'error': f"Model {model_type} not found in pipeline."}

    row = {}
    for col in _feature_cols:
        val = form.get(col, '')
        if val == '':
            row[col] = np.nan
        else:
            try:
                row[col] = float(val)
            except ValueError:
                row[col] = val

    import pandas as pd
    X_input = pd.DataFrame([row], columns=_feature_cols)

    try:
        salary_pred  = float(rf_reg_model.predict(X_input)[0])
        salary_pred  = round(max(3.0, min(salary_pred, 30.0)), 2)
    except Exception as e:
        salary_pred = 0.0

    result_dict = {
        'model_type': model_type,
        'salary': salary_pred,
        'error': None
    }

    if model_type == 'softmax':
        # Predict Probabilities for 3 classes
        probs = clf_model.predict_proba(X_input)[0]
        classes = clf_model.classes_
        
        # Match probabilities to their class labels
        prob_dict = {str(c): round(p * 100, 1) for c, p in zip(classes, probs)}
        
        # Also get the top prediction
        top_class = clf_model.predict(X_input)[0]
        
        result_dict['softmax'] = True
        result_dict['top_class'] = top_class
        result_dict['probs'] = prob_dict
    else:
        # Binary Classification
        placed_prob  = clf_model.predict_proba(X_input)[0][1]
        placed_class = int(clf_model.predict(X_input)[0])
            
        result_dict['placed'] = placed_class
        result_dict['placed_prob'] = round(placed_prob * 100, 1)

    return result_dict

def get_form_dropdown_values(df):
    return {
        'gender_vals': sorted(df['Gender'].dropna().unique().tolist()),
        'city_vals': sorted(df['City'].dropna().unique().tolist()),
        'college_tier_vals': sorted(df['CollegeTier'].dropna().unique().tolist()),
        'stream_vals': sorted(df['Stream'].dropna().unique().tolist()),
        'spec_vals': sorted(df['Specialisation'].dropna().unique().tolist()),
        'hostel_vals': ['No', 'Yes'],
        'backlog_vals': ['No', 'Yes']
    }

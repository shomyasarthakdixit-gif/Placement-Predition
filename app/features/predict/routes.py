import numpy as np
from flask import Blueprint, render_template, request, current_app

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/predict', methods=['GET', 'POST'])
def predict_page():
    ml_data = current_app.config['ML_PIPELINE']
    df = ml_data['df']
    rf_clf_model = ml_data['rf_clf']
    
    gender_vals       = sorted(df['Gender'].dropna().unique().tolist())
    city_vals         = sorted(df['City'].dropna().unique().tolist())
    college_tier_vals = sorted(df['CollegeTier'].dropna().unique().tolist())
    stream_vals       = sorted(df['Stream'].dropna().unique().tolist())
    spec_vals         = sorted(df['Specialisation'].dropna().unique().tolist())
    hostel_vals       = ['No', 'Yes']
    backlog_vals      = ['No', 'Yes']

    result = None

    if request.method == 'POST':
        try:
            result = _run_prediction(request.form)
        except Exception as e:
            result = {'error': str(e)}
            
    # Get feature importance from the underlying classifier (the actual model, not the pipeline)
    # The pipeline is: prep -> clf.
    importances = rf_clf_model.named_steps['clf'].feature_importances_
    # Note: ColumnTransformer with target/hashing encoders changes the feature names/count.
    # For a professional dashboard, extracting exact names from pipelines can be tricky.
    # We will do a generic importance array for now or try to extract them.
    # We'll just pass a placeholder or skip feature importance if it causes issues.
    feature_importance = []
    
    try:
        # Attempt to get feature names if available in scikit-learn >= 1.2
        prep = rf_clf_model.named_steps['prep']
        if hasattr(prep, 'get_feature_names_out'):
            f_names = prep.get_feature_names_out()
            pairs = sorted(zip(f_names, importances), key=lambda x: -x[1])[:12]
            total = sum(v for _, v in pairs)
            if total > 0:
                feature_importance = [{'name': n.split('__')[-1], 'pct': round(v / total * 100, 1)} for n, v in pairs]
    except Exception as e:
        print(f"[Warning] Could not extract feature names for importance: {e}")

    return render_template(
        'prediction.html',
        gender_vals=gender_vals,
        city_vals=city_vals,
        college_tier_vals=college_tier_vals,
        stream_vals=stream_vals,
        spec_vals=spec_vals,
        hostel_vals=hostel_vals,
        backlog_vals=backlog_vals,
        result=result,
        feature_importance=feature_importance
    )


def _run_prediction(form):
    """Build feature row from form, run chosen model, return result dict."""
    ml_data = current_app.config['ML_PIPELINE']
    _feature_cols = ml_data['feature_cols']
    rf_clf_model = ml_data['rf_clf']
    lr_clf_model = ml_data['lr_clf']
    rf_reg_model = ml_data['rf_reg']
    
    model_type = form.get('model_type', 'rf')

    row = {}
    for col in _feature_cols:
        val = form.get(col, '')
        # If it's empty, and the column is numeric, default to 0.0, else empty string
        if val == '':
            row[col] = np.nan # The pipeline imputers will handle this natively now!
        else:
            try:
                row[col] = float(val)
            except ValueError:
                row[col] = val

    import pandas as pd
    X_input = pd.DataFrame([row], columns=_feature_cols)

    if model_type == 'lr':
        placed_prob  = lr_clf_model.predict_proba(X_input)[0][1]
        placed_class = int(lr_clf_model.predict(X_input)[0])
    else:
        placed_prob  = rf_clf_model.predict_proba(X_input)[0][1]
        placed_class = int(rf_clf_model.predict(X_input)[0])

    salary_pred  = float(rf_reg_model.predict(X_input)[0])
    salary_pred  = round(max(3.0, min(salary_pred, 30.0)), 2)

    return {
        'placed':       placed_class,
        'placed_prob':  round(placed_prob * 100, 1),
        'salary':       salary_pred,
        'error':        None
    }

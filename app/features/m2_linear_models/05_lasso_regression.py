from flask import render_template, request, current_app, url_for
from . import m2_bp
from app.features.utils import run_prediction_for_model, get_form_dropdown_values

@m2_bp.route('/lasso_regression', methods=['GET', 'POST'])
def lasso_regression_page():
    ml_data = current_app.config['ML_PIPELINE']
    df = ml_data['df']
    
    dropdowns = get_form_dropdown_values(df)
    result = None
    
    if request.method == 'POST':
        result = run_prediction_for_model(request.form, 'lasso')
        
    return render_template(
        'model_classification.html',
        model_title="Lasso Classifier (L1 Penalty)",
        model_desc="Logistic regression with L1 regularization for feature selection.",
        action_url=url_for('m2_linear_models.lasso_regression_page'),
        result=result,
        **dropdowns
    )

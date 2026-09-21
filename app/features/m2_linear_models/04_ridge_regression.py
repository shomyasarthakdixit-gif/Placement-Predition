from flask import render_template, request, current_app, url_for
from . import m2_bp
from app.features.utils import run_prediction_for_model, get_form_dropdown_values

@m2_bp.route('/ridge_regression', methods=['GET', 'POST'])
def ridge_regression_page():
    ml_data = current_app.config['ML_PIPELINE']
    df = ml_data['df']
    
    dropdowns = get_form_dropdown_values(df)
    result = None
    
    if request.method == 'POST':
        result = run_prediction_for_model(request.form, 'ridge')
        
    return render_template(
        'model_classification.html',
        model_title="Ridge Classifier (L2 Penalty)",
        model_desc="Logistic regression with L2 regularization to prevent overfitting.",
        action_url=url_for('m2_linear_models.ridge_regression_page'),
        result=result,
        **dropdowns
    )

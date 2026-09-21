from flask import render_template, request, current_app, url_for
from . import m2_bp
from app.features.utils import run_prediction_for_model, get_form_dropdown_values

@m2_bp.route('/elastic_net', methods=['GET', 'POST'])
def elastic_net_page():
    ml_data = current_app.config['ML_PIPELINE']
    df = ml_data['df']
    
    dropdowns = get_form_dropdown_values(df)
    result = None
    
    if request.method == 'POST':
        result = run_prediction_for_model(request.form, 'elasticnet')
        
    return render_template(
        'model_classification.html',
        model_title="Elastic Net Classifier (L1 + L2 Penalty)",
        model_desc="Logistic regression combining both Lasso and Ridge regularization.",
        action_url=url_for('m2_linear_models.elastic_net_page'),
        result=result,
        **dropdowns
    )

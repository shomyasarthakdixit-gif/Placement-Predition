from flask import render_template, request, current_app, url_for
from . import m3_bp
from app.features.utils import run_prediction_for_model, get_form_dropdown_values

@m3_bp.route('/decision_tree', methods=['GET', 'POST'])
def decision_tree_page():
    ml_data = current_app.config['ML_PIPELINE']
    df = ml_data['df']
    dropdowns = get_form_dropdown_values(df)
    result = None
    if request.method == 'POST':
        result = run_prediction_for_model(request.form, 'dt')
        
    return render_template(
        'model_classification.html',
        model_title="Decision Tree Classifier",
        model_desc="Tree-based model with non-linear decision boundaries.",
        action_url=url_for('m3_tree_models.decision_tree_page'),
        result=result,
        **dropdowns
    )

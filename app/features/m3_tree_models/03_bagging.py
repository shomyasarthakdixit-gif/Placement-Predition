from flask import render_template, request, current_app, url_for
from . import m3_bp
from app.features.utils import run_prediction_for_model, get_form_dropdown_values

@m3_bp.route('/bagging', methods=['GET', 'POST'])
def bagging_page():
    ml_data = current_app.config['ML_PIPELINE']
    df = ml_data['df']
    
    dropdowns = get_form_dropdown_values(df)
    result = None
    
    if request.method == 'POST':
        result = run_prediction_for_model(request.form, 'bag')
        
    return render_template(
        'model_classification.html',
        model_title="Bagging Classifier",
        model_desc="Bootstrap Aggregating (Bagging) builds multiple independent decision trees on random subsets of data and averages their predictions to reduce variance and prevent overfitting.",
        action_url=url_for('m3_tree_models.bagging_page'),
        result=result,
        **dropdowns
    )

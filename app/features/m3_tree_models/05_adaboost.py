from flask import render_template, request, current_app, url_for
from . import m3_bp
from app.features.utils import run_prediction_for_model, get_form_dropdown_values

@m3_bp.route('/adaboost', methods=['GET', 'POST'])
def adaboost_page():
    ml_data = current_app.config['ML_PIPELINE']
    df = ml_data['df']
    
    dropdowns = get_form_dropdown_values(df)
    result = None
    
    if request.method == 'POST':
        result = run_prediction_for_model(request.form, 'ada')
        
    return render_template(
        'model_classification.html',
        model_title="AdaBoost Classifier",
        model_desc="Adaptive Boosting fits a sequence of weak learners on repeatedly modified versions of the data, paying more attention to training instances that were previously misclassified.",
        action_url=url_for('m3_tree_models.adaboost_page'),
        result=result,
        **dropdowns
    )

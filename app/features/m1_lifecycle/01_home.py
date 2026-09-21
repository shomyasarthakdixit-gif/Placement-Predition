from flask import render_template, current_app
from . import m1_bp

@m1_bp.route('/')
def home():
    ml_data = current_app.config['ML_PIPELINE']
    df = ml_data['df']
    numeric_cols = ml_data['numeric_cols']
    categorical_cols = ml_data['categorical_cols']
    
    return render_template(
        'index.html',
        num_cols=len(df.columns),
        numeric_count=len(numeric_cols),
        cat_count=len(categorical_cols),
        columns=df.columns.tolist()
    )

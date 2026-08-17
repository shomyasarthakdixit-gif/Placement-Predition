from flask import Blueprint, render_template, current_app

home_bp = Blueprint('home', __name__)

@home_bp.route('/')
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

@home_bp.route('/load_data')
def load_page():
    ml_data = current_app.config['ML_PIPELINE']
    df = ml_data['df']
    data_html = df.head(100).to_html(classes="data-table", index=False, border=0)
    return render_template('load_data.html', data_html=data_html)

@home_bp.route('/full_data')
def full_data_page():
    ml_data = current_app.config['ML_PIPELINE']
    df = ml_data['df']
    data_html = df.to_html(classes="data-table", index=False, border=0)
    return render_template('full_data.html', data_html=data_html)

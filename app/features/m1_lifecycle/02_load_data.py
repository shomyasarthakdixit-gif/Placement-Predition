from flask import render_template, current_app
from . import m1_bp

@m1_bp.route('/load_data')
def load_page():
    ml_data = current_app.config['ML_PIPELINE']
    df = ml_data['df']
    data_html = df.head(100).to_html(classes="data-table", index=False, border=0)
    return render_template('load_data.html', data_html=data_html)

@m1_bp.route('/full_data')
def full_data_page():
    ml_data = current_app.config['ML_PIPELINE']
    df = ml_data['df']
    data_html = df.to_html(classes="data-table", index=False, border=0)
    return render_template('full_data.html', data_html=data_html)

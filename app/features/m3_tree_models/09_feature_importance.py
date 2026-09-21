from flask import render_template
from . import m3_bp

@m3_bp.route('/feature_importance')
def feature_importance_page():
    return f"<h2>feature_importance</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

from flask import render_template
from . import m2_bp

@m2_bp.route('/linear_regression')
def linear_regression_page():
    return f"<h2>linear_regression</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

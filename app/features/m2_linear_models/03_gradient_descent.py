from flask import render_template
from . import m2_bp

@m2_bp.route('/gradient_descent')
def gradient_descent_page():
    return f"<h2>gradient_descent</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

from flask import render_template
from . import m4_bp

@m4_bp.route('/elbow_method')
def elbow_method_page():
    return f"<h2>elbow_method</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

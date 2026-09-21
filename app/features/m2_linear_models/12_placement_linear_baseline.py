from flask import render_template
from . import m2_bp

@m2_bp.route('/placement_linear_baseline')
def placement_linear_baseline_page():
    return f"<h2>placement_linear_baseline</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

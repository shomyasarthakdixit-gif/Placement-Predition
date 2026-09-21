from flask import render_template
from . import m2_bp

@m2_bp.route('/missing_value_handling')
def missing_value_handling_page():
    return f"<h2>missing_value_handling</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

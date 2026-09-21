from flask import render_template
from . import m2_bp

@m2_bp.route('/normal_equation')
def normal_equation_page():
    return f"<h2>normal_equation</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

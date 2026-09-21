from flask import render_template
from . import m3_bp

@m3_bp.route('/bagging')
def bagging_page():
    return f"<h2>bagging</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

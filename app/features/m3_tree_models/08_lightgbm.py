from flask import render_template
from . import m3_bp

@m3_bp.route('/lightgbm')
def lightgbm_page():
    return f"<h2>lightgbm</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

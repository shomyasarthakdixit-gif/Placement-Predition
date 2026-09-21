from flask import render_template
from . import m4_bp

@m4_bp.route('/kmeans_plus_plus')
def kmeans_plus_plus_page():
    return f"<h2>kmeans_plus_plus</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

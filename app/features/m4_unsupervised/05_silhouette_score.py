from flask import render_template
from . import m4_bp

@m4_bp.route('/silhouette_score')
def silhouette_score_page():
    return f"<h2>silhouette_score</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

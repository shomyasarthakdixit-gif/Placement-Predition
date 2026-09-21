from flask import render_template
from . import m4_bp

@m4_bp.route('/dendrogram')
def dendrogram_page():
    return f"<h2>dendrogram</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

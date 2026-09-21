from flask import render_template
from . import m4_bp

@m4_bp.route('/mini_batch_kmeans')
def mini_batch_kmeans_page():
    return f"<h2>mini_batch_kmeans</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

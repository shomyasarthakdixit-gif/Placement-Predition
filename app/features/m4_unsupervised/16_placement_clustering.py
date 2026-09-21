from flask import render_template
from . import m4_bp

@m4_bp.route('/placement_clustering')
def placement_clustering_page():
    return f"<h2>placement_clustering</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

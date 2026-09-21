from flask import render_template
from . import m4_bp

@m4_bp.route('/hierarchical_clustering')
def hierarchical_clustering_page():
    return f"<h2>hierarchical_clustering</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

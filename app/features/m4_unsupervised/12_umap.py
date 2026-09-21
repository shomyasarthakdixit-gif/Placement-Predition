from flask import render_template
from . import m4_bp

@m4_bp.route('/umap')
def umap_page():
    return f"<h2>umap</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

from flask import render_template
from . import m3_bp

@m3_bp.route('/placement_tree_models')
def placement_tree_models_page():
    return f"<h2>placement_tree_models</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

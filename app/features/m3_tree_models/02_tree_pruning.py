from flask import render_template
from . import m3_bp

@m3_bp.route('/tree_pruning')
def tree_pruning_page():
    return f"<h2>tree_pruning</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

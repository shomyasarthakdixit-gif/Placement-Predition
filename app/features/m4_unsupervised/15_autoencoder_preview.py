from flask import render_template
from . import m4_bp

@m4_bp.route('/autoencoder_preview')
def autoencoder_preview_page():
    return f"<h2>autoencoder_preview</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

from flask import render_template
from . import m4_bp

@m4_bp.route('/linkages')
def linkages_page():
    return f"<h2>linkages</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

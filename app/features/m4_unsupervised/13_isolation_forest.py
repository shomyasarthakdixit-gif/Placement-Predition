from flask import render_template
from . import m4_bp

@m4_bp.route('/isolation_forest')
def isolation_forest_page():
    return f"<h2>isolation_forest</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

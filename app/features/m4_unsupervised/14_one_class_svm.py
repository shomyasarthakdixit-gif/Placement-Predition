from flask import render_template
from . import m4_bp

@m4_bp.route('/one_class_svm')
def one_class_svm_page():
    return f"<h2>one_class_svm</h2><p>This module is part of the syllabus but currently has no interactive demo.</p>"

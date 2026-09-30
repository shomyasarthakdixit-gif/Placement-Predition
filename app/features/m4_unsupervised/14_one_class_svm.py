import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import render_template_string, current_app
from . import m4_bp

@m4_bp.route('/one_class_svm')
def one_class_svm_page():
    ml_data = current_app.config.get('ML_PIPELINE', {})
    if not ml_data:
        return "ML Pipeline not loaded.", 500
        
    ocsvm = ml_data.get('ocsvm')
    tsne_comps = ml_data.get('tsne_comps')
    X_sample = ml_data.get('X_sample')
    
    plot_url = ""
    if tsne_comps is not None and ocsvm is not None:
        labels = ocsvm.predict(X_sample)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#1e1e2f')
        ax.patch.set_facecolor('#1e1e2f')
        
        # 1 = inlier, -1 = outlier
        colors = ['#f59e0b' if l == -1 else '#3b82f6' for l in labels]
        
        ax.scatter(tsne_comps[:, 0], tsne_comps[:, 1], color=colors, alpha=0.5, s=15)
        ax.set_title("One-Class SVM Anomaly Detection (t-SNE Space)", fontweight='bold')
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Normal (Inlier)', markerfacecolor='#3b82f6', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Anomaly (Outlier)', markerfacecolor='#f59e0b', markersize=8)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        
    template = '''
    {% extends "base.html" %}
    {% block title %}One-Class SVM{% endblock %}
    {% block content %}
    <div class="card">
        <h2>One-Class SVM (Novelty Detection)</h2>
        <p>One-Class Support Vector Machine is an algorithm used for anomaly and novelty detection. It learns a decision boundary that encompasses the "normal" data points in a high-dimensional feature space, marking anything outside that boundary as an anomaly.</p>
        <p>Because OCSVM scales poorly with the number of samples (O(N^2) to O(N^3)), this visualization is run on the representative 2,000 row sample, plotted against t-SNE coordinates.</p>
        <div style="text-align: center; margin-top: 20px;">
            <img src="data:image/png;base64,''' + plot_url + '''" style="max-width:100%; border-radius:10px;">
        </div>
        <div style="margin-top: 30px; text-align: center;">
            <a href="javascript:history.back()" class="btn btn-secondary"><i class="ph ph-arrow-left"></i> Back to Previous</a>
        </div>
    </div>
    {% endblock %}
    '''
    return render_template_string(template)

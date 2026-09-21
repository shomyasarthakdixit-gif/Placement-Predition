import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import render_template_string, current_app
from . import m4_bp

@m4_bp.route('/pca')
def pca_page():
    ml_data = current_app.config.get('ML_PIPELINE', {})
    if not ml_data:
        return "ML Pipeline not loaded.", 500
        
    kmeans = ml_data.get('kmeans')
    pca_comps = ml_data.get('pca_comps')
    
    plot_url = ""
    if pca_comps is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#1e1e2f')
        ax.patch.set_facecolor('#1e1e2f')
        
        scatter = ax.scatter(pca_comps[:, 0], pca_comps[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.5, s=10)
        ax.set_title("Student Segmentation (PCA reduced, K-Means Clustered)", fontweight='bold')
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        fig.colorbar(scatter, ax=ax, label="Cluster")
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        
    template = '''
    {% extends "base.html" %}
    {% block title %}PCA Visualization{% endblock %}
    {% block content %}
    <div class="card">
        <h2>Principal Component Analysis (PCA)</h2>
        <p>This scatter plot visualizes the dataset reduced to 2 principal components. The colors represent K-Means clusters.</p>
        <img src="data:image/png;base64,''' + plot_url + '''" style="max-width:100%; border-radius:10px;">
    </div>
    {% endblock %}
    '''
    return render_template_string(template)

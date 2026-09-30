import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import render_template_string, current_app
from . import m4_bp

@m4_bp.route('/mini_batch_kmeans')
def mini_batch_kmeans_page():
    ml_data = current_app.config.get('ML_PIPELINE', {})
    if not ml_data:
        return "ML Pipeline not loaded.", 500
        
    mbk = ml_data.get('mbk')
    pca_comps = ml_data.get('pca_comps')
    
    plot_url = ""
    if pca_comps is not None and mbk is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#1e1e2f')
        ax.patch.set_facecolor('#1e1e2f')
        
        scatter = ax.scatter(pca_comps[:, 0], pca_comps[:, 1], c=mbk.labels_, cmap='plasma', alpha=0.5, s=10)
        ax.set_title("Student Segmentation (PCA reduced, MiniBatch K-Means Clustered)", fontweight='bold')
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
    {% block title %}Mini-Batch K-Means Visualization{% endblock %}
    {% block content %}
    <div class="card">
        <h2>Mini-Batch K-Means</h2>
        <p>Mini-Batch K-Means is a faster alternative to standard K-Means. It uses small random batches of data to update cluster centroids, making it highly scalable for massive datasets.</p>
        <p>Below is the result of Mini-Batch K-Means plotted against the PCA-reduced dataset.</p>
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

import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import render_template_string, current_app
from . import m4_bp

@m4_bp.route('/dbscan')
def dbscan_page():
    ml_data = current_app.config.get('ML_PIPELINE', {})
    if not ml_data:
        return "ML Pipeline not loaded.", 500
        
    dbscan_labels = ml_data.get('dbscan_labels')
    tsne_comps = ml_data.get('tsne_comps')
    
    plot_url = ""
    if tsne_comps is not None and dbscan_labels is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#1e1e2f')
        ax.patch.set_facecolor('#1e1e2f')
        
        scatter = ax.scatter(tsne_comps[:, 0], tsne_comps[:, 1], c=dbscan_labels, cmap='Set1', alpha=0.6, s=15)
        ax.set_title("DBSCAN Clustering (t-SNE reduced space)", fontweight='bold')
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        fig.colorbar(scatter, ax=ax, label="Cluster (Noise = -1)")
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        
    template = '''
    {% extends "base.html" %}
    {% block title %}DBSCAN Visualization{% endblock %}
    {% block content %}
    <div class="card">
        <h2>Density-Based Spatial Clustering of Applications with Noise (DBSCAN)</h2>
        <p>Unlike K-Means, DBSCAN does not assume clusters are spherical and does not require specifying the number of clusters in advance. It groups together points that are closely packed together, marking as outliers (noise) points that lie alone in low-density regions.</p>
        <p>Because pairwise distance calculations are computationally expensive on 50,000 rows, this plot represents a representative random sample of 2,000 students projected into a 2D space using t-SNE.</p>
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

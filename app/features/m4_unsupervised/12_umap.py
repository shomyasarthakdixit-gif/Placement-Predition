import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import render_template_string, current_app
from . import m4_bp

@m4_bp.route('/umap')
def umap_page():
    ml_data = current_app.config.get('ML_PIPELINE', {})
    if not ml_data:
        return "ML Pipeline not loaded.", 500
        
    umap_comps = ml_data.get('umap_comps')
    
    plot_url = ""
    if umap_comps is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#1e1e2f')
        ax.patch.set_facecolor('#1e1e2f')
        
        ax.scatter(umap_comps[:, 0], umap_comps[:, 1], color='#a855f7', alpha=0.5, s=10)
        ax.set_title("UMAP Representation of Students", fontweight='bold')
        ax.set_xlabel("UMAP Component 1")
        ax.set_ylabel("UMAP Component 2")
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        
    template = '''
    {% extends "base.html" %}
    {% block title %}UMAP Visualization{% endblock %}
    {% block content %}
    <div class="card">
        <h2>Uniform Manifold Approximation and Projection (UMAP)</h2>
        <p>UMAP is a modern, fast dimension reduction technique that is highly effective at preserving both local and global data structure, making it generally superior to t-SNE for many clustering tasks.</p>
        <p>Like t-SNE, this visualization was performed on a representative 2,000 row sample to ensure responsive computation.</p>
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

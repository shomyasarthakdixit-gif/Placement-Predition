import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import render_template_string, current_app
from . import m4_bp

@m4_bp.route('/tsne')
def tsne_page():
    ml_data = current_app.config.get('ML_PIPELINE', {})
    if not ml_data:
        return "ML Pipeline not loaded.", 500
        
    tsne_comps = ml_data.get('tsne_comps')
    
    plot_url = ""
    if tsne_comps is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#1e1e2f')
        ax.patch.set_facecolor('#1e1e2f')
        
        ax.scatter(tsne_comps[:, 0], tsne_comps[:, 1], color='#3b82f6', alpha=0.5, s=10)
        ax.set_title("t-SNE Representation of Students", fontweight='bold')
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        
    template = '''
    {% extends "base.html" %}
    {% block title %}t-SNE Visualization{% endblock %}
    {% block content %}
    <div class="card">
        <h2>t-Distributed Stochastic Neighbor Embedding (t-SNE)</h2>
        <p>t-SNE is a statistical method for visualizing high-dimensional data by giving each datapoint a location in a two or three-dimensional map. It is highly effective at clustering local neighborhoods of similar data points.</p>
        <p>Because t-SNE is extremely computationally intensive (O(N^2) complexity), this visualization was performed on a random sample of 2,000 students to ensure reasonable rendering times.</p>
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

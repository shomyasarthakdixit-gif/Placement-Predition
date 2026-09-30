import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import render_template_string, current_app
from . import m4_bp

@m4_bp.route('/isolation_forest')
def isolation_forest_page():
    ml_data = current_app.config.get('ML_PIPELINE', {})
    if not ml_data:
        return "ML Pipeline not loaded.", 500
        
    iso = ml_data.get('iso')
    pca_comps = ml_data.get('pca_comps')
    
    plot_url = ""
    if pca_comps is not None and iso is not None:
        labels = iso.predict(ml_data.get('X_trans'))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#1e1e2f')
        ax.patch.set_facecolor('#1e1e2f')
        
        # 1 = inlier, -1 = outlier
        colors = ['#ef4444' if l == -1 else '#3b82f6' for l in labels]
        
        ax.scatter(pca_comps[:, 0], pca_comps[:, 1], color=colors, alpha=0.5, s=10)
        ax.set_title("Isolation Forest Anomaly Detection", fontweight='bold')
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        
        # Add legend manually
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Normal (Inlier)', markerfacecolor='#3b82f6', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Anomaly (Outlier)', markerfacecolor='#ef4444', markersize=8)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        
    template = '''
    {% extends "base.html" %}
    {% block title %}Isolation Forest{% endblock %}
    {% block content %}
    <div class="card">
        <h2>Isolation Forest (Anomaly Detection)</h2>
        <p>Isolation Forest detects anomalies by randomly selecting a feature and randomly selecting a split value. Since anomalies are sparse and different, they are isolated closer to the root of the tree (shorter path lengths) compared to normal points.</p>
        <p>Below we visualize the detected anomalies (red) in the student dataset, projected onto 2D PCA space.</p>
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

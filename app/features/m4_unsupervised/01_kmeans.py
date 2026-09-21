import numpy as np
from flask import render_template_string, current_app
from sklearn.metrics import silhouette_score
from . import m4_bp

@m4_bp.route('/kmeans')
def kmeans_page():
    ml_data = current_app.config.get('ML_PIPELINE', {})
    if not ml_data:
        return "ML Pipeline not loaded.", 500
        
    kmeans = ml_data.get('kmeans')
    X_trans = ml_data.get('X_trans')
    
    silhouette = 0
    if X_trans is not None and kmeans is not None:
        subset_idx = np.random.choice(X_trans.shape[0], min(2000, X_trans.shape[0]), replace=False)
        silhouette = silhouette_score(X_trans[subset_idx], kmeans.labels_[subset_idx])
        silhouette = round(silhouette, 3)
        
    template = '''
    {% extends "base.html" %}
    {% block title %}K-Means Clustering{% endblock %}
    {% block content %}
    <div class="card">
        <h2>K-Means Clustering Evaluation</h2>
        <div class="result-metrics" style="margin-top:20px;">
            <div class="result-metric">
                <span class="metric-val">''' + str(silhouette) + '''</span>
                <span class="metric-lbl">Silhouette Score</span>
            </div>
            <div class="result-metric">
                <span class="metric-val">4</span>
                <span class="metric-lbl">Number of Clusters (K)</span>
            </div>
        </div>
        <p style="margin-top:20px;">Silhouette Score measures how similar an object is to its own cluster compared to other clusters. Values range from -1 to 1.</p>
    </div>
    {% endblock %}
    '''
    return render_template_string(template)

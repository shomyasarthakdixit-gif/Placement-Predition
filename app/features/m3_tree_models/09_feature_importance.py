import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import render_template_string, current_app
from . import m3_bp

@m3_bp.route('/feature_importance')
def feature_importance_page():
    ml_data = current_app.config.get('ML_PIPELINE', {})
    if not ml_data:
        return "ML Pipeline not loaded.", 500
        
    rf_clf = ml_data.get('rf_clf')
    preprocessor = ml_data.get('preprocessor')
    
    plot_url = ""
    if rf_clf is not None and preprocessor is not None:
        try:
            # Extract feature names from the preprocessor if possible
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"Feature {i}" for i in range(len(rf_clf.named_steps['clf'].feature_importances_))]
            
        importances = rf_clf.named_steps['clf'].feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Take top 15 features to avoid a massive cluttered plot
        top_n = min(15, len(indices))
        top_indices = indices[:top_n]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#1e1e2f')
        ax.patch.set_facecolor('#1e1e2f')
        
        bars = ax.barh(range(top_n), top_importances[::-1], color='#3b82f6', align='center')
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features[::-1])
        ax.set_xlabel("Relative Importance")
        ax.set_title("Random Forest Top Feature Importances", fontweight='bold')
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        
    template = '''
    {% extends "base.html" %}
    {% block title %}Feature Importance{% endblock %}
    {% block content %}
    <div class="card" style="margin-bottom: 24px; border-left: 4px solid var(--accent-primary);">
        <h1 style="margin-bottom: 10px;"><i class="ph ph-chart-bar"></i> Feature Importance (Gini)</h1>
        <p style="color: var(--text-secondary); font-size: 1.1rem; line-height: 1.5;">Understanding which factors influence placement predictions the most.</p>
    </div>
    
    <div class="card">
        <h2>Global Feature Importance</h2>
        <p>Tree-based models inherently calculate feature importance during training. The importance of a feature is computed as the (normalized) total reduction of the criterion (e.g., Gini Impurity) brought by that feature. It is also known as the Gini Importance.</p>
        <p>Below is the importance derived from our <strong>Random Forest Classifier</strong> trained on the 50,000 student dataset.</p>
        <div style="text-align: center; margin-top: 20px;">
            <img src="data:image/png;base64,''' + plot_url + '''" style="max-width:100%; border-radius:10px;">
        </div>
    </div>
    
    <div style="margin-top: 30px; margin-bottom: 30px; text-align: center;">
        <a href="javascript:history.back()" class="btn btn-secondary"><i class="ph ph-arrow-left"></i> Back to Previous</a>
    </div>
    {% endblock %}
    '''
    return render_template_string(template)

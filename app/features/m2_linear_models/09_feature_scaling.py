import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import render_template, request, current_app
from . import m2_bp

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', transparent=False)
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return encoded

@m2_bp.route('/scaling', methods=['GET', 'POST'])
def feature_scaling_page():
    ml_data = current_app.config['ML_PIPELINE']
    df = ml_data['df']
    _numeric_cols = ml_data['numeric_cols']
    
    dynamic_plot = None
    scaled_html = None
    metrics_html = None
    
    if request.method == 'POST':
        col = request.form.get('fe_column')
        scaler_type = request.form.get('fe_scaler')
        
        if col and pd.api.types.is_numeric_dtype(df[col]):
            from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
            
            raw_data = df[[col]].dropna()
            
            if scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                scaler = None
                
            if scaler:
                scaled_data = scaler.fit_transform(raw_data)
                
                metrics = pd.DataFrame({
                    'Metric': ['Min', 'Max', 'Mean', 'Std Dev', 'Median', 'IQR'],
                    'Raw (Before)': [
                        raw_data[col].min(), raw_data[col].max(), raw_data[col].mean(), 
                        raw_data[col].std(), raw_data[col].median(), 
                        raw_data[col].quantile(0.75) - raw_data[col].quantile(0.25)
                    ],
                    'Scaled (After)': [
                        scaled_data.min(), scaled_data.max(), scaled_data.mean(), 
                        scaled_data.std(), np.median(scaled_data), 
                        np.percentile(scaled_data, 75) - np.percentile(scaled_data, 25)
                    ]
                }).round(4)
                
                metrics_html = metrics.to_html(classes="data-table", index=False)
                
                sample_df = pd.DataFrame({
                    'Raw Value': raw_data[col].head(10).values,
                    f'{scaler_type.capitalize()} Scaled': scaled_data[:10].flatten()
                }).round(4)
                scaled_html = sample_df.to_html(classes="data-table", index=False)
                
                plt.style.use('dark_background')
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.patch.set_facecolor('#1e1e2f')
                axes[0].patch.set_facecolor('#1e1e2f')
                axes[1].patch.set_facecolor('#1e1e2f')
                sns.kdeplot(data=raw_data, x=col, fill=True, color='#0f3460', ax=axes[0])
                axes[0].set_title(f'Before Scaling ({col})', fontweight='bold')
                
                sns.kdeplot(scaled_data.flatten(), fill=True, color='#1a7fcf', ax=axes[1])
                axes[1].set_title(f'After {scaler_type.capitalize()} Scaling', fontweight='bold')
                axes[1].set_xlabel('Scaled Value')
                
                plt.tight_layout()
                dynamic_plot = _fig_to_b64(fig)
                
    return render_template('scaling.html', 
                           numeric_cols=_numeric_cols,
                           dynamic_plot=dynamic_plot,
                           metrics_html=metrics_html,
                           scaled_html=scaled_html)

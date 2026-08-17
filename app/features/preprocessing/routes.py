import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Blueprint, render_template, request, current_app

preprocessing_bp = Blueprint('preprocessing', __name__)

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', transparent=True)
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return encoded

@preprocessing_bp.route('/scaling', methods=['GET', 'POST'])
def scaling_page():
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
                
                # Plot Original vs Scaled Distributions
                plt.style.use('dark_background')
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.patch.set_alpha(0.0)
                axes[0].patch.set_alpha(0.0)
                axes[1].patch.set_alpha(0.0)
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

@preprocessing_bp.route('/feature_engg')
def feature_engg_page():
    return render_template('feature_engg.html')

@preprocessing_bp.route('/encoding', methods=['GET', 'POST'])
def encoding_page():
    ml_data = current_app.config['ML_PIPELINE']
    df = ml_data['df']
    _categorical_cols = ml_data['categorical_cols']
    
    dynamic_plot = None
    encoded_html = None
    
    if request.method == 'POST':
        col = request.form.get('fe_column')
        encoder_type = request.form.get('fe_encoder')
        
        if col and col in _categorical_cols:
            import category_encoders as ce
            # Some categorical columns might have missing values, let's fill them with 'Missing' for the demo
            raw_data = df[[col]].fillna('Missing').astype(str).copy()
            y = df['PlacementStatus'] if 'PlacementStatus' in df.columns else None
            
            if encoder_type == 'ohe':
                encoder = ce.OneHotEncoder(cols=[col], use_cat_names=True)
            elif encoder_type == 'ordinal':
                encoder = ce.OrdinalEncoder(cols=[col])
            elif encoder_type == 'target':
                encoder = ce.TargetEncoder(cols=[col])
            elif encoder_type == 'hashing':
                encoder = ce.HashingEncoder(cols=[col], n_components=4)
            else:
                encoder = None
                
            if encoder:
                if encoder_type == 'target':
                    encoded_data = encoder.fit_transform(raw_data, y)
                else:
                    encoded_data = encoder.fit_transform(raw_data)
                
                # Add suffix to encoded columns to avoid name collisions
                encoded_data_renamed = encoded_data.add_suffix('_encoded')
                
                # Sample Table (Top 15)
                sample_df = pd.concat([raw_data, encoded_data_renamed], axis=1).head(15)
                encoded_html = sample_df.to_html(classes="data-table", index=False)
                
                # Plot
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_alpha(0.0)
                ax.patch.set_alpha(0.0)
                if encoder_type == 'ohe':
                    counts = encoded_data_renamed.sum().sort_values(ascending=False).head(10)
                    sns.barplot(x=counts.values, y=counts.index, palette='viridis', ax=ax)
                    ax.set_title(f'One-Hot Encoding Expansion (Top 10 categories in {col})', fontweight='bold')
                    ax.set_xlabel('Count in Dataset')
                elif encoder_type == 'ordinal':
                    # Drop duplicates to show mapping
                    enc_col = f"{col}_encoded"
                    mapping = pd.concat([raw_data, encoded_data_renamed], axis=1).drop_duplicates().sort_values(by=col).head(15)
                    sns.barplot(data=mapping, x=col, y=enc_col, palette='magma', ax=ax)
                    ax.set_title(f'Ordinal Encoding Mapping for {col} (Sample)', fontweight='bold')
                    ax.set_ylabel('Ordinal Value')
                    plt.xticks(rotation=45, ha='right')
                elif encoder_type == 'target':
                    enc_col = f"{col}_encoded"
                    mapping = pd.concat([raw_data, encoded_data_renamed], axis=1).drop_duplicates().sort_values(by=enc_col, ascending=False).head(15)
                    sns.barplot(data=mapping, x=enc_col, y=col, palette='coolwarm', ax=ax)
                    ax.set_title(f'Target Encoding Mapping (Placement Rate) for {col}', fontweight='bold')
                    ax.set_xlabel('Target Encoded Value')
                elif encoder_type == 'hashing':
                    hash_sums = encoded_data_renamed.sum()
                    sns.barplot(x=hash_sums.index, y=hash_sums.values, palette='plasma', ax=ax)
                    ax.set_title(f'Hashing Encoding Distribution ({col} -> 4 Buckets)', fontweight='bold')
                    ax.set_ylabel('Count')
                
                plt.tight_layout()
                dynamic_plot = _fig_to_b64(fig)
                
    return render_template('encoding.html', 
                           categorical_cols=_categorical_cols,
                           dynamic_plot=dynamic_plot,
                           encoded_html=encoded_html)


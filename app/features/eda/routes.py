import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Blueprint, render_template, request, send_from_directory, current_app

eda_bp = Blueprint('eda', __name__)
_TARGET_CLASS = 'PlacementStatus'

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight', transparent=True)
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return encoded

@eda_bp.route('/eda', methods=['GET', 'POST'])
def eda_page():
    ml_data = current_app.config['ML_PIPELINE']
    df = ml_data['df']
    _numeric_cols = ml_data['numeric_cols']
    _categorical_cols = ml_data['categorical_cols']
    
    PLOT_PATH = os.path.join(os.path.dirname(current_app.root_path), "Output", "plot")
    
    static_plots = []
    if os.path.exists(PLOT_PATH):
        static_plots = sorted([f for f in os.listdir(PLOT_PATH) if f.lower().endswith('.png')])

    dynamic_plot = None
    eda_output   = None

    _static_op_map = {
        'static_histogram':       'Histogram.png',
        'static_piechart':        'PieChart.png',
        'static_placementstatus': 'PlacementStatusCount.png',
        'static_scatter':         'ScatterPlot.png',
        'static_boxplot':         'BoxPlot.png',
        'static_dotplot':         'DotPlot.png',
        'static_gendercount':     'GendervsPlacementCount.png',
        'static_correlation':     'CorrelationMatrix.png',
        'static_pairplot':        'PairPlot.png',
    }

    if request.method == 'POST':
        operation = request.form.get('operation', '')

        if operation in _static_op_map:
            fname = _static_op_map[operation]
            fpath = os.path.join(PLOT_PATH, fname)
            if os.path.exists(fpath):
                with open(fpath, 'rb') as f:
                    dynamic_plot = base64.b64encode(f.read()).decode('utf-8')
            else:
                eda_output = f'<p style="color:#c0392b;">Plot {fname} not found.</p>'

        elif operation == 'summary_stats':
            eda_output = df.describe().T.to_html(classes="data-table")

        elif operation == 'missing_values':
            mv = df.isnull().sum().to_frame(name='Missing Values')
            mv['% Missing'] = (mv['Missing Values'] / len(df) * 100).round(2)
            eda_output = mv.to_html(classes="data-table")

        elif operation == 'value_counts':
            col = request.form.get('col_vc')
            if col:
                eda_output = df[col].value_counts().to_frame(name='Count').to_html(classes="data-table")

        elif operation == 'outlier_stats':
            col = request.form.get('col_outlier')
            if col and pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                q1, q2, q3 = series.quantile(0.25), series.median(), series.quantile(0.75)
                iqr = q3 - q1
                lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers = series[(series < lb) | (series > ub)]
                
                stats_df = pd.DataFrame({
                    'Metric': ['Min', 'Max', 'Q1 (25%)', 'Q2 (Median)', 'Q3 (75%)', 'IQR', 'Lower Bound (LB)', 'Upper Bound (UB)', 'Outliers Count'],
                    'Value': [series.min(), series.max(), q1, q2, q3, iqr, lb, ub, len(outliers)]
                })
                stats_df['Value'] = stats_df['Value'].apply(lambda x: round(x, 4) if isinstance(x, float) else x)
                
                outlier_vals_str = "None"
                if len(outliers) > 0:
                    outlier_vals = outliers.head(10).apply(lambda x: round(x, 4) if isinstance(x, float) else x).values
                    outlier_vals_str = ", ".join(map(str, outlier_vals))
                    if len(outliers) > 10:
                        outlier_vals_str += f" ... (and {len(outliers)-10} more)"
                
                stats_df.loc[len(stats_df)] = ['Sample Outliers', outlier_vals_str]
                eda_output = f"<h3 style='margin-bottom: 12px;'>Outlier Analysis for <strong>{col}</strong></h3>" + stats_df.to_html(classes="data-table", index=False)

        elif operation == 'kde_plot':
            col = request.form.get('col_kde')
            if col and pd.api.types.is_numeric_dtype(df[col]):
                fig, ax = plt.subplots(figsize=(9, 5))
                sns.kdeplot(data=df, x=col, fill=True, color='#0f3460', ax=ax)
                ax.set_title(f'KDE Plot — {col}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                dynamic_plot = _fig_to_b64(fig)

        elif operation == 'violin_plot':
            col_x = request.form.get('col_violin_x')
            col_y = request.form.get('col_violin_y')
            if col_x and col_y and pd.api.types.is_numeric_dtype(df[col_y]):
                fig, ax = plt.subplots(figsize=(9, 5))
                sns.violinplot(data=df, x=col_x, y=col_y, palette='muted', ax=ax)
                ax.set_title(f'Violin Plot — {col_y} by {col_x}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                dynamic_plot = _fig_to_b64(fig)

        elif operation == 'mv_heatmap':
            num_df = df.select_dtypes(include='number').drop(columns=['StudentID', 'IsAnomaly'], errors='ignore')
            fig, ax = plt.subplots(figsize=(14, 11))
            mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
            sns.heatmap(num_df.corr(), mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, linewidths=0.5, annot_kws={'size': 7}, ax=ax)
            ax.set_title('Full Correlation Heatmap', fontsize=14, fontweight='bold', pad=16)
            plt.tight_layout()
            dynamic_plot = _fig_to_b64(fig)

        elif operation == 'mv_pairplot':
            cols = request.form.getlist('mv_pairplot_cols')
            if len(cols) >= 2:
                sample = df[cols + [_TARGET_CLASS]].dropna().sample(min(2000, len(df)), random_state=42)
                sample[_TARGET_CLASS] = sample[_TARGET_CLASS].map({0: 'Not Placed', 1: 'Placed'})
                g = sns.pairplot(sample, hue=_TARGET_CLASS, palette={'Placed': '#1a7fcf', 'Not Placed': '#e74c3c'}, plot_kws={'alpha': 0.45, 's': 18}, diag_kind='kde')
                g.fig.patch.set_alpha(0.0)
                for ax in g.axes.flatten():
                    ax.patch.set_alpha(0.0)
                dynamic_plot = _fig_to_b64(g.fig)
            else:
                eda_output = '<p style="color:#c0392b;">Please select at least 2 numeric columns.</p>'

        elif operation == 'mv_grouped_bar':
            col_x, col_hue = request.form.get('mv_bar_x'), request.form.get('mv_bar_hue')
            if col_x and col_hue:
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_alpha(0.0)
                ax.patch.set_alpha(0.0)
                sns.countplot(data=df, x=col_x, hue=col_hue, palette='Set2', ax=ax)
                ax.set_title(f'Grouped Bar Chart — {col_x} by {col_hue}', fontsize=14, fontweight='bold')
                plt.xticks(rotation=30, ha='right'); plt.tight_layout()
                dynamic_plot = _fig_to_b64(fig)

        elif operation == 'mv_boxgroup':
            col_num, col_grp, col_hue = request.form.get('mv_box_num'), request.form.get('mv_box_grp'), request.form.get('mv_box_hue')
            if col_num and col_grp and pd.api.types.is_numeric_dtype(df[col_num]):
                fig, ax = plt.subplots(figsize=(11, 6))
                sns.boxplot(data=df, x=col_grp, y=col_num, hue=col_hue, palette='Set3', ax=ax)
                plt.xticks(rotation=30, ha='right'); plt.tight_layout()
                dynamic_plot = _fig_to_b64(fig)

        elif operation == 'mv_strip':
            col_x, col_y, col_hue = request.form.get('mv_strip_x'), request.form.get('mv_strip_y'), request.form.get('mv_strip_hue')
            if col_x and col_y and pd.api.types.is_numeric_dtype(df[col_y]):
                sample = df.sample(min(3000, len(df)), random_state=1)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.stripplot(data=sample, x=col_x, y=col_y, hue=col_hue, palette='husl', jitter=True, size=4, alpha=0.65, ax=ax)
                plt.xticks(rotation=30, ha='right'); plt.tight_layout()
                dynamic_plot = _fig_to_b64(fig)

        elif operation == 'mv_facet':
            col_num, col_row = request.form.get('mv_facet_num'), request.form.get('mv_facet_row')
            if col_num and col_row and pd.api.types.is_numeric_dtype(df[col_num]):
                g = sns.FacetGrid(df, col=col_row, col_wrap=3, height=3.5, sharey=False)
                g.map(sns.histplot, col_num, bins=25, color='#1a7fcf', edgecolor='white')
                plt.tight_layout()
                dynamic_plot = _fig_to_b64(g.fig)

    return render_template('eda.html',
                           static_plots=static_plots,
                           dynamic_plot=dynamic_plot,
                           eda_output=eda_output,
                           numeric_cols=_numeric_cols,
                           categorical_cols=_categorical_cols,
                           all_cols=df.columns.tolist())

@eda_bp.route('/plots/<filename>')
def serve_plot(filename):
    PLOT_PATH = os.path.join(os.path.dirname(current_app.root_path), "Output", "plot")
    return send_from_directory(PLOT_PATH, filename)

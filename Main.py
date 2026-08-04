import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data")
PLOT_PATH = os.path.join(BASE_DIR, "Output", "plot")

# Load data once at startup
df = pd.read_csv(os.path.join(DATA_PATH, "placement_predict_50k Dataset (2).csv"))

# Pre-compute column info for the home page
_numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
_categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()


# ── Routes ──────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template(
        'index.html',
        num_cols=len(df.columns),
        numeric_count=len(_numeric_cols),
        cat_count=len(_categorical_cols),
        columns=df.columns.tolist()
    )


@app.route('/load_data')
def load_page():
    data_html = df.head(100).to_html(classes="data-table", index=False, border=0)
    return render_template('load_data.html', data_html=data_html)


@app.route('/full_data')
def full_data_page():
    data_html = df.to_html(classes="data-table", index=False, border=0)
    return render_template('full_data.html', data_html=data_html)


@app.route('/eda', methods=['GET', 'POST'])
def eda_page():
    # Collect static plots
    static_plots = []
    if os.path.exists(PLOT_PATH):
        static_plots = sorted(
            [f for f in os.listdir(PLOT_PATH) if f.lower().endswith('.png')]
        )

    dynamic_plot = None
    eda_output = None

    if request.method == 'POST':
        operation = request.form.get('operation')

        if operation == 'summary_stats':
            eda_output = df.describe().T.to_html(classes="data-table")

        elif operation == 'missing_values':
            mv = df.isnull().sum().to_frame(name='Missing Values')
            mv['% Missing'] = (mv['Missing Values'] / len(df) * 100).round(2)
            eda_output = mv.to_html(classes="data-table")

        elif operation == 'value_counts':
            col = request.form.get('col_vc')
            if col:
                eda_output = df[col].value_counts().to_frame(name='Count').to_html(classes="data-table")

        elif operation == 'kde_plot':
            col = request.form.get('col_kde')
            if col and pd.api.types.is_numeric_dtype(df[col]):
                fig, ax = plt.subplots(figsize=(9, 5))
                sns.kdeplot(data=df, x=col, fill=True, color='#0f3460', ax=ax)
                ax.set_title(f'KDE Plot — {col}', fontsize=14, fontweight='bold')
                ax.set_xlabel(col)
                ax.set_ylabel('Density')
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

    return render_template(
        'eda.html',
        static_plots=static_plots,
        dynamic_plot=dynamic_plot,
        eda_output=eda_output,
        numeric_cols=_numeric_cols,
        categorical_cols=_categorical_cols,
        all_cols=df.columns.tolist()
    )


@app.route('/plots/<filename>')
def serve_plot(filename):
    return send_from_directory(PLOT_PATH, filename)


@app.route('/feature_engg')
def feature_engg_page():
    return render_template('feature_engg.html')


# ── Helpers ─────────────────────────────────────────────────────────────

def _fig_to_b64(fig):
    """Convert a matplotlib figure to a base64 PNG string and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return encoded


# ── Entry point ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

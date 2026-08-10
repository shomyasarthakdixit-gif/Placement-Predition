import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, send_from_directory, jsonify

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Data")
PLOT_PATH = os.path.join(BASE_DIR, "Output", "plot")

# ── Load data once at startup ────────────────────────────────────────────
df = pd.read_csv(os.path.join(DATA_PATH, "placement_predict_50k Dataset (2).csv"))

# Pre-compute column info
_numeric_cols      = df.select_dtypes(include=['number']).columns.tolist()
_categorical_cols  = df.select_dtypes(exclude=['number']).columns.tolist()

# Remove target / ID cols from features used for prediction
_TARGET_CLASS  = 'PlacementStatus'
_TARGET_REG    = 'Salary Package'
_DROP_COLS     = ['StudentID', 'IsAnomaly', 'CGPA_Tier', _TARGET_CLASS, _TARGET_REG]

# ── Train ML models at startup ──────────────────────────────────────────
clf_model  = None   # RandomForestClassifier → PlacementStatus
reg_model  = None   # RandomForestRegressor  → Salary Package
_label_encoders = {}
_feature_cols   = []

def _train_models():
    global clf_model, reg_model, _label_encoders, _feature_cols
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder

    feat_df = df.drop(columns=_DROP_COLS, errors='ignore').copy()
    _feature_cols = feat_df.columns.tolist()

    # Encode categoricals
    for col in feat_df.select_dtypes(exclude=['number']).columns:
        le = LabelEncoder()
        feat_df[col] = le.fit_transform(feat_df[col].astype(str))
        _label_encoders[col] = le

    X = feat_df.values
    y_cls = df[_TARGET_CLASS].values
    y_reg = df.loc[df[_TARGET_CLASS] == 1, _TARGET_REG].values
    X_reg = feat_df.loc[df[_TARGET_CLASS] == 1].values

    # Classification model
    clf_model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf_model.fit(X, y_cls)

    # Regression model (only placed students)
    reg_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    reg_model.fit(X_reg, y_reg)

    print("[ML] Models trained successfully.")

_train_models()


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
    eda_output   = None

    # Map static plot operations → filename in Output/plot
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

        # ── Static plot display ──────────────────────────────────
        if operation in _static_op_map:
            fname = _static_op_map[operation]
            fpath = os.path.join(PLOT_PATH, fname)
            if os.path.exists(fpath):
                with open(fpath, 'rb') as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                dynamic_plot = encoded
            else:
                eda_output = (
                    f'<p style="color:#c0392b;">Plot file <code>{fname}</code> not found. '
                    f'Run <code>Source/EDA.py</code> first.</p>'
                )

        # ── Summary stats ────────────────────────────────────────
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

        # ── Outlier Statistics ───────────────────────────────────
        elif operation == 'outlier_stats':
            col = request.form.get('col_outlier')
            if col and pd.api.types.is_numeric_dtype(df[col]):
                series = df[col].dropna()
                q1 = series.quantile(0.25)
                q2 = series.median()
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lb = q1 - 1.5 * iqr
                ub = q3 + 1.5 * iqr
                outliers = series[(series < lb) | (series > ub)]
                
                stats_df = pd.DataFrame({
                    'Metric': ['Min', 'Max', 'Q1 (25%)', 'Q2 (Median)', 'Q3 (75%)', 'IQR', 'Lower Bound (LB)', 'Upper Bound (UB)', 'Outliers Count'],
                    'Value': [series.min(), series.max(), q1, q2, q3, iqr, lb, ub, len(outliers)]
                })
                
                # Format to a specific decimal limit for neatness
                stats_df['Value'] = stats_df['Value'].apply(lambda x: round(x, 4) if isinstance(x, float) else x)
                
                outlier_vals_str = "None"
                if len(outliers) > 0:
                    outlier_vals = outliers.head(10).apply(lambda x: round(x, 4) if isinstance(x, float) else x).values
                    outlier_vals_str = ", ".join(map(str, outlier_vals))
                    if len(outliers) > 10:
                        outlier_vals_str += f" ... (and {len(outliers)-10} more)"
                
                # Append sample outliers row
                stats_df.loc[len(stats_df)] = ['Sample Outliers', outlier_vals_str]
                
                eda_output = f"<h3 style='margin-bottom: 12px;'>Outlier Analysis for <strong>{col}</strong></h3>" + stats_df.to_html(classes="data-table", index=False)

        # ── KDE Plot ─────────────────────────────────────────────
        elif operation == 'kde_plot':
            col = request.form.get('col_kde')
            if col and pd.api.types.is_numeric_dtype(df[col]):
                fig, ax = plt.subplots(figsize=(9, 5))
                sns.kdeplot(data=df, x=col, fill=True, color='#0f3460', ax=ax)
                ax.set_title(f'KDE Plot — {col}', fontsize=14, fontweight='bold')
                ax.set_xlabel(col); ax.set_ylabel('Density')
                plt.tight_layout()
                dynamic_plot = _fig_to_b64(fig)

        # ── Violin Plot ──────────────────────────────────────────
        elif operation == 'violin_plot':
            col_x = request.form.get('col_violin_x')
            col_y = request.form.get('col_violin_y')
            if col_x and col_y and pd.api.types.is_numeric_dtype(df[col_y]):
                fig, ax = plt.subplots(figsize=(9, 5))
                sns.violinplot(data=df, x=col_x, y=col_y, palette='muted', ax=ax)
                ax.set_title(f'Violin Plot — {col_y} by {col_x}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                dynamic_plot = _fig_to_b64(fig)

        # ── MULTIVARIATE: Full Correlation Heatmap ───────────────
        elif operation == 'mv_heatmap':
            num_df = df.select_dtypes(include='number').drop(
                columns=['StudentID', 'IsAnomaly'], errors='ignore')
            fig, ax = plt.subplots(figsize=(14, 11))
            mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
            sns.heatmap(
                num_df.corr(), mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, linewidths=0.5,
                annot_kws={'size': 7}, ax=ax
            )
            ax.set_title('Full Correlation Heatmap (Numeric Features)',
                         fontsize=14, fontweight='bold', pad=16)
            plt.tight_layout()
            dynamic_plot = _fig_to_b64(fig)

        # ── MULTIVARIATE: Custom Pairplot ────────────────────────
        elif operation == 'mv_pairplot':
            cols = request.form.getlist('mv_pairplot_cols')
            if len(cols) >= 2:
                sample = df[cols + [_TARGET_CLASS]].dropna().sample(
                    min(2000, len(df)), random_state=42)
                sample[_TARGET_CLASS] = sample[_TARGET_CLASS].map(
                    {0: 'Not Placed', 1: 'Placed'})
                g = sns.pairplot(sample, hue=_TARGET_CLASS,
                                 palette={'Placed': '#1a7fcf', 'Not Placed': '#e74c3c'},
                                 plot_kws={'alpha': 0.45, 's': 18}, diag_kind='kde')
                g.fig.suptitle('Pairplot of Selected Features', y=1.02,
                               fontsize=13, fontweight='bold')
                dynamic_plot = _fig_to_b64(g.fig)
            else:
                eda_output = '<p style="color:#c0392b;">Please select at least 2 numeric columns.</p>'

        # ── MULTIVARIATE: Grouped Bar Chart ──────────────────────
        elif operation == 'mv_grouped_bar':
            col_x   = request.form.get('mv_bar_x')
            col_hue = request.form.get('mv_bar_hue')
            if col_x and col_hue:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.countplot(data=df, x=col_x, hue=col_hue,
                              palette='Set2', ax=ax)
                ax.set_title(f'Grouped Bar Chart — {col_x} by {col_hue}',
                             fontsize=14, fontweight='bold')
                ax.set_xlabel(col_x); ax.set_ylabel('Count')
                plt.xticks(rotation=30, ha='right')
                plt.tight_layout()
                dynamic_plot = _fig_to_b64(fig)

        # ── MULTIVARIATE: Box Plot by Group ─────────────────────
        elif operation == 'mv_boxgroup':
            col_num = request.form.get('mv_box_num')
            col_grp = request.form.get('mv_box_grp')
            col_hue = request.form.get('mv_box_hue') or None
            if col_num and col_grp and pd.api.types.is_numeric_dtype(df[col_num]):
                fig, ax = plt.subplots(figsize=(11, 6))
                sns.boxplot(data=df, x=col_grp, y=col_num,
                            hue=col_hue, palette='Set3', ax=ax)
                title = f'Box Plot — {col_num} by {col_grp}'
                if col_hue:
                    title += f' (hue: {col_hue})'
                ax.set_title(title, fontsize=14, fontweight='bold')
                plt.xticks(rotation=30, ha='right')
                plt.tight_layout()
                dynamic_plot = _fig_to_b64(fig)

        # ── MULTIVARIATE: Strip / Swarm Plot ─────────────────────
        elif operation == 'mv_strip':
            col_x = request.form.get('mv_strip_x')
            col_y = request.form.get('mv_strip_y')
            col_hue = request.form.get('mv_strip_hue') or None
            if col_x and col_y and pd.api.types.is_numeric_dtype(df[col_y]):
                sample = df.sample(min(3000, len(df)), random_state=1)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.stripplot(data=sample, x=col_x, y=col_y,
                              hue=col_hue, palette='husl',
                              jitter=True, size=4, alpha=0.65, ax=ax)
                title = f'Strip Plot — {col_y} by {col_x}'
                if col_hue:
                    title += f' (hue: {col_hue})'
                ax.set_title(title, fontsize=14, fontweight='bold')
                plt.xticks(rotation=30, ha='right')
                plt.tight_layout()
                dynamic_plot = _fig_to_b64(fig)

        # ── MULTIVARIATE: Facet Grid ─────────────────────────────
        elif operation == 'mv_facet':
            col_num = request.form.get('mv_facet_num')
            col_row = request.form.get('mv_facet_row')
            if col_num and col_row and pd.api.types.is_numeric_dtype(df[col_num]):
                g = sns.FacetGrid(df, col=col_row, col_wrap=3,
                                  height=3.5, sharey=False)
                g.map(sns.histplot, col_num, bins=25,
                      color='#1a7fcf', edgecolor='white')
                g.set_titles(col_template=f'{col_row}: {{col_name}}')
                g.fig.suptitle(f'Facet Grid — {col_num} by {col_row}',
                               y=1.03, fontsize=13, fontweight='bold')
                plt.tight_layout()
                dynamic_plot = _fig_to_b64(g.fig)

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


# ── Prediction Route ─────────────────────────────────────────────────────

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    # Build distinct values for dropdowns
    gender_vals       = sorted(df['Gender'].dropna().unique().tolist())
    city_vals         = sorted(df['City'].dropna().unique().tolist())
    college_tier_vals = sorted(df['CollegeTier'].dropna().unique().tolist())
    stream_vals       = sorted(df['Stream'].dropna().unique().tolist())
    spec_vals         = sorted(df['Specialisation'].dropna().unique().tolist())
    hostel_vals       = ['No', 'Yes']
    backlog_vals      = ['No', 'Yes']

    result = None

    if request.method == 'POST':
        try:
            result = _run_prediction(request.form)
        except Exception as e:
            result = {'error': str(e)}

    return render_template(
        'prediction.html',
        gender_vals=gender_vals,
        city_vals=city_vals,
        college_tier_vals=college_tier_vals,
        stream_vals=stream_vals,
        spec_vals=spec_vals,
        hostel_vals=hostel_vals,
        backlog_vals=backlog_vals,
        result=result,
        feature_importance=_get_feature_importance()
    )


def _run_prediction(form):
    """Build feature row from form, run both models, return result dict."""
    from sklearn.preprocessing import LabelEncoder

    row = {}
    for col in _feature_cols:
        if col in _label_encoders:
            # Categorical: encode via saved LabelEncoder
            val = form.get(col, '')
            le  = _label_encoders[col]
            if val in le.classes_:
                row[col] = le.transform([val])[0]
            else:
                row[col] = 0
        else:
            # Numeric
            try:
                row[col] = float(form.get(col, 0))
            except (ValueError, TypeError):
                row[col] = 0.0

    X_input = np.array([[row[c] for c in _feature_cols]])

    # Classification
    placed_prob  = clf_model.predict_proba(X_input)[0][1]
    placed_class = int(clf_model.predict(X_input)[0])

    # Regression (always run, but show only if placed)
    salary_pred  = float(reg_model.predict(X_input)[0])
    salary_pred  = round(max(3.0, min(salary_pred, 30.0)), 2)

    return {
        'placed':       placed_class,
        'placed_prob':  round(placed_prob * 100, 1),
        'salary':       salary_pred,
        'error':        None
    }


def _get_feature_importance():
    """Return top-12 feature importances for the classifier."""
    if clf_model is None:
        return []
    importances = clf_model.feature_importances_
    pairs = sorted(zip(_feature_cols, importances), key=lambda x: -x[1])[:12]
    total = sum(v for _, v in pairs)
    return [{'name': n, 'pct': round(v / total * 100, 1)} for n, v in pairs]


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

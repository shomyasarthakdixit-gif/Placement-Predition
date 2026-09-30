from flask import render_template
from . import m2_bp

@m2_bp.route('/feature_scaling')
def feature_scaling_page():
    content = r"""
    <p>Feature scaling is a critical preprocessing step for many machine learning algorithms, especially those that rely on distance metrics (like K-Means) or gradient descent (like Logistic Regression and Neural Networks). It ensures that all features contribute proportionately to the final prediction and speeds up model convergence.</p>
    <p>Without scaling, a feature with a range of [0, 100,000] would completely dominate a feature with a range of [0, 1].</p>
    """
    
    math_content = r"""
    <p><strong>1. Standard Scaler (Z-score Normalization)</strong></p>
    <p style="text-align:center;">\\[ z = \\frac{x - \\mu}{\\sigma} \\]</p>
    <p>Centers the data at 0 with a standard deviation of 1. Used in our pipeline for relatively clean, normally distributed numeric columns like CGPA.</p>
    <hr style="border: 0.5px solid #334155; margin: 20px 0;">
    
    <p><strong>2. Min-Max Scaler</strong></p>
    <p style="text-align:center;">\\[ X_{scaled} = \\frac{X - X_{min}}{X_{max} - X_{min}} \\]</p>
    <p>Scales the data to a fixed range, usually [0, 1]. Used in our pipeline for features strictly bounded like Aptitude and Coding Test Scores.</p>
    <hr style="border: 0.5px solid #334155; margin: 20px 0;">
    
    <p><strong>3. Robust Scaler</strong></p>
    <p style="text-align:center;">\\[ X_{scaled} = \\frac{X - Q_1}{Q_3 - Q_1} \\]</p>
    <p>Uses statistics that are robust to outliers (the interquartile range). Applied to features that may have extreme spikes.</p>
    """
    
    application_content = r"""
    <p>In <code>ml_pipeline.py</code>, we utilize a <code>ColumnTransformer</code> to apply these specific scalers selectively based on the column profile:</p>
    <ul>
        <li><strong>StandardScaler</strong> is applied to <code>CGPA</code> and <code>AttendancePercent</code>.</li>
        <li><strong>MinMaxScaler</strong> is applied to <code>AptitudeTestScore</code> and <code>CodingTestScore</code>.</li>
        <li><strong>RobustScaler</strong> is used as the default for the remaining continuous variables that may be prone to outliers.</li>
    </ul>
    """


    from flask import url_for
    img_url = url_for('m1_lifecycle.serve_plot', filename='FeatureScaling.png')
    application_content += f'''
    <div style="background-color: var(--bg-secondary); padding: 15px; border-radius: 8px; margin-top: 20px; text-align: center;">
        <h4 style="margin-top: 0; text-align: left;">Algorithm Output Visualised</h4>
        <img src="{img_url}" style="max-width: 100%; border-radius: 8px; border: 1px solid var(--border-color);" alt="FeatureScaling.png">
    </div>
    '''
    return render_template(
        'educational_concept.html',
        title="Feature Scaling",
        subtitle="Bringing features to a common scale for optimal model performance.",
        icon="ph ph-arrows-out",
        content=content,
        math_content=math_content,
        application_content=application_content
    )

from flask import render_template
from . import m2_bp

@m2_bp.route('/missing_value_handling')
def missing_value_handling_page():
    content = r"""
    <p>Real-world datasets like our student placement data are rarely perfect. They often contain missing values due to surveying errors, optional fields, or data extraction glitches.</p>
    <p>Scikit-Learn estimators generally crash if they encounter <code>NaN</code> (Not a Number) values, so Imputation (the process of replacing missing data with substituted values) is a mandatory preprocessing step.</p>
    """
    
    math_content = r"""
    <p><strong>1. Mean Imputation</strong></p>
    <p>Replaces missing numeric values with the mathematical average of the column. This is fast and simple but can be heavily skewed by extreme outliers.</p>
    <hr style="border: 0.5px solid #334155; margin: 20px 0;">
    
    <p><strong>2. Median Imputation</strong></p>
    <p>Replaces missing numeric values with the middle value of the sorted column. This is statistically robust to outliers.</p>
    <hr style="border: 0.5px solid #334155; margin: 20px 0;">
    
    <p><strong>3. Most Frequent (Mode) Imputation</strong></p>
    <p>Replaces missing categorical or nominal values with the category that appears the most often in that column.</p>
    <hr style="border: 0.5px solid #334155; margin: 20px 0;">
    
    <p><strong>4. Missing Indicators</strong></p>
    <p>Creates a brand new binary column (0 or 1) that acts as a flag indicating whether the data was missing. Sometimes, the fact that data is missing is a predictive signal in itself (e.g., hiding a bad CGPA).</p>
    """
    
    application_content = r"""
    <p>In our <code>ml_pipeline.py</code>, we utilize the <code>SimpleImputer</code> class strategically before scaling or encoding:</p>
    <ul>
        <li><strong>Numeric (Clean):</strong> We use <code>strategy='mean'</code> and <code>add_indicator=True</code> for standard numbers, capturing the missingness signal.</li>
        <li><strong>Numeric (Robust/MinMax):</strong> We use <code>strategy='median'</code> to prevent outliers from distorting the imputation.</li>
        <li><strong>Categorical (All types):</strong> We use <code>strategy='most_frequent'</code> because you cannot calculate an average of text labels like "Male" or "Female".</li>
    </ul>
    """

    return render_template(
        'educational_concept.html',
        title="Missing Value Handling",
        subtitle="Strategies for imputing gaps in the dataset to prevent model crashes.",
        icon="ph ph-detective",
        content=content,
        math_content=math_content,
        application_content=application_content
    )

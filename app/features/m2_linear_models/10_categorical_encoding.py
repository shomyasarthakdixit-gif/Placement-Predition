from flask import render_template
from . import m2_bp

@m2_bp.route('/categorical_encoding')
def categorical_encoding_page():
    content = r"""
    <p>Machine Learning models require numerical input. Categorical Encoding is the process of transforming text-based or discrete category labels into numeric representations so that the model can process them mathematically.</p>
    <p>Because our placement dataset is extremely rich with non-numeric data (such as Specialisation, City, Stream, Gender), picking the correct encoder for each feature prevents data sparsity and preserves the relationships inside the data.</p>
    """
    
    math_content = r"""
    <p><strong>1. One-Hot Encoding (OHE)</strong></p>
    <p>Converts a single category into multiple binary columns (0s and 1s). Perfect for low-cardinality nominal variables without intrinsic order (e.g., Gender).</p>
    <p style="text-align:center;"><em>A \\( \\rightarrow \\) [1, 0, 0], B \\( \\rightarrow \\) [0, 1, 0]</em></p>
    <hr style="border: 0.5px solid #334155; margin: 20px 0;">
    
    <p><strong>2. Ordinal Encoding</strong></p>
    <p>Assigns sequential integers based on inherent order (e.g., CollegeTier: Tier 1 \\( \\rightarrow 1 \\), Tier 2 \\( \\rightarrow 2 \\)). Preserves the magnitude relationship.</p>
    <hr style="border: 0.5px solid #334155; margin: 20px 0;">
    
    <p><strong>3. Target Encoding</strong></p>
    <p>Replaces a categorical value with the mean of the target variable for that category. Very powerful for high-cardinality features (like City or Specialisation) without blowing up the column count.</p>
    <hr style="border: 0.5px solid #334155; margin: 20px 0;">
    
    <p><strong>4. Hashing Encoding</strong></p>
    <p>Applies a hash function to map categories to an arbitrary integer array of fixed size. Useful for high-cardinality features where Target Encoding might overfit.</p>
    """
    
    from flask import url_for
    img_url = url_for('m1_lifecycle.serve_plot', filename='CategoricalEncoding.png')
    
    application_content = f"""
    <p>In our advanced <code>ml_pipeline.py</code>, we use <code>category_encoders</code> to intelligently apply the optimal method:</p>
    <ul>
        <li><strong>OrdinalEncoder:</strong> <code>CollegeTier</code> (Tier 1 vs Tier 2 has a distinct mathematical ranking).</li>
        <li><strong>OneHotEncoder:</strong> <code>Gender</code>, <code>Hostel</code>, <code>HistoryOfBacklogs</code> (Low cardinality).</li>
        <li><strong>TargetEncoder:</strong> <code>City</code>, <code>Specialisation</code> (High cardinality, correlates directly with Placement chance).</li>
        <li><strong>HashingEncoder:</strong> <code>Stream</code> (To manage the massive variance of engineering/degree streams).</li>
    </ul>
    
    <div style="background-color: var(--bg-secondary); padding: 15px; border-radius: 8px; margin-top: 20px;">
        <h4 style="margin-top: 0;">Example Output (Before vs After)</h4>
        <div style="overflow-x: auto;">
            <table class="data-table" style="width: 100%; border-collapse: collapse; font-size: 0.85rem; text-align: left;">
                <thead>
                    <tr style="border-bottom: 1px solid var(--border-color);">
                        <th>Raw 'Gender'</th>
                        <th>Encoded 'Gender_M'</th>
                        <th>Encoded 'Gender_F'</th>
                        <th style="border-left: 2px solid var(--border-color); padding-left: 10px;">Raw 'CollegeTier'</th>
                        <th>Encoded 'CollegeTier'</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>Male</td><td>1.0</td><td>0.0</td><td style="border-left: 2px solid var(--border-color); padding-left: 10px;">Tier 1</td><td>1.0</td></tr>
                    <tr><td>Female</td><td>0.0</td><td>1.0</td><td style="border-left: 2px solid var(--border-color); padding-left: 10px;">Tier 2</td><td>2.0</td></tr>
                    <tr><td>Male</td><td>1.0</td><td>0.0</td><td style="border-left: 2px solid var(--border-color); padding-left: 10px;">Tier 2</td><td>2.0</td></tr>
                </tbody>
            </table>
        </div>
        <p style="font-size: 0.85rem; margin-top: 10px; color: var(--text-secondary);">
            <em>* Notice how One-Hot Encoding created multiple binary columns, whereas Ordinal Encoding preserved the single column but converted it to a numerical rank.</em>
        </p>
    </div>
    
    <div style="background-color: var(--bg-secondary); padding: 15px; border-radius: 8px; margin-top: 20px; text-align: center;">
        <h4 style="margin-top: 0; text-align: left;">Target Encoding Visualised</h4>
        <p style="font-size: 0.9rem; text-align: left; margin-bottom: 15px;">By replacing the high-cardinality "Specialisation" labels with the average placement probability of that group, the ML model receives a powerful continuous numeric predictor.</p>
        <img src="{img_url}" style="max-width: 100%; border-radius: 8px; border: 1px solid var(--border-color);" alt="Target Encoding Plot">
    </div>
    """

    return render_template(
        'educational_concept.html',
        title="Categorical Encoding",
        subtitle="Converting text categories into model-ready numbers.",
        icon="ph ph-translate",
        content=content,
        math_content=math_content,
        application_content=application_content
    )

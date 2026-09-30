from flask import render_template
from . import m4_bp

@m4_bp.route('/elbow_method')
def elbow_method_page():
    content = r"""
    <p>When using K-Means clustering, the number of clusters (\\( K \\)) must be specified manually. The <strong>Elbow Method</strong> is a heuristic used to determine the optimal number of clusters for a dataset.</p>
    <p>By plotting the sum of squared distances from each point to its assigned center (Inertia or WCSS) against different values of \\( K \\), we can find the point where adding more clusters no longer yields a significant decrease in variance—the "elbow" of the curve.</p>
    """
    
    math_content = r"""
    <p><strong>Within-Cluster Sum of Squares (WCSS)</strong></p>
    <p style="text-align:center;">\\[ WCSS = \\sum_{j=1}^{k} \\sum_{i=1}^{n_j} (x_{ij} - C_j)^2 \\]</p>
    <p>Where:</p>
    <ul>
        <li>\\( k \\) is the number of clusters.</li>
        <li>\\( n_j \\) is the number of points in cluster \\( j \\).</li>
        <li>\\( x_{ij} \\) is the \\( i \\)-th point in cluster \\( j \\).</li>
        <li>\\( C_j \\) is the centroid of cluster \\( j \\).</li>
    </ul>
    """
    
    application_content = r"""
    <p>For our Student Placement Prediction project, calculating the Elbow Method dynamically on 50,000 rows across multiple values of \\( K \\) (e.g., 1 to 10) takes a considerable amount of time.</p>
    <p>During our offline Exploratory Data Analysis, the Elbow Method indicated that \\( K=4 \\) was the optimal split. Therefore, in our <code>ml_pipeline.py</code>, we explicitly set <code>n_clusters=4</code> for K-Means to segment students into roughly: High-Performers, Average-Tech, Average-Non-Tech, and At-Risk students.</p>
    """


    from flask import url_for
    img_url = url_for('m1_lifecycle.serve_plot', filename='ElbowMethod.png')
    application_content += f'''
    <div style="background-color: var(--bg-secondary); padding: 15px; border-radius: 8px; margin-top: 20px; text-align: center;">
        <h4 style="margin-top: 0; text-align: left;">Algorithm Output Visualised</h4>
        <img src="{img_url}" style="max-width: 100%; border-radius: 8px; border: 1px solid var(--border-color);" alt="ElbowMethod.png">
    </div>
    '''
    return render_template(
        'educational_concept.html',
        title="Elbow Method",
        subtitle="Finding the optimal number of clusters heuristically.",
        icon="ph ph-chart-line",
        content=content,
        math_content=math_content,
        application_content=application_content
    )

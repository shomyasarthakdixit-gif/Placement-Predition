from flask import render_template
from . import m2_bp

@m2_bp.route('/linear_regression')
def linear_regression_page():
    content = r"""
    <p>Linear Regression is one of the foundational algorithms in machine learning. It models the relationship between a dependent variable (target) and one or more independent variables (features) by fitting a linear equation to observed data.</p>
    <p>In our placement prediction project, we use linear regression concepts extensively as the basis for logistic regression, which we use to classify whether a student is placed or not.</p>
    """
    
    math_content = r"""
    <p>The hypothesis function for linear regression with multiple features is defined as:</p>
    <p style="text-align:center; font-size: 1.2rem;">\\[ h_\\theta(x) = \\theta_0 + \\theta_1 x_1 + \\theta_2 x_2 + \\dots + \\theta_n x_n \\]</p>
    <p>Where:</p>
    <ul>
        <li>\\( h_\\theta(x) \\) is the predicted value.</li>
        <li>\\( \\theta_0 \\) is the y-intercept (bias term).</li>
        <li>\\( \\theta_1, \\theta_2, \\dots, \\theta_n \\) are the model weights for each feature.</li>
        <li>\\( x_1, x_2, \\dots, x_n \\) are the feature values for a given instance.</li>
    </ul>
    """
    
    application_content = r"""
    <p>While standard linear regression is used for continuous numeric predictions (like salary), we adapt this linear approach using a sigmoid function (Logistic Regression) to predict binary outcomes (Placed vs Not Placed).</p>
    <p>Our model evaluates factors such as CGPA, Aptitude Test Scores, and Internships by assigning a specific weight (\\( \\theta \\)) to each feature. If a feature strongly increases the chance of placement (e.g., high CGPA), its weight will be highly positive.</p>
    """
    from flask import url_for
    img_url = url_for('m1_lifecycle.serve_plot', filename='LinearRegression.png')
    application_content += f'''
    <div style="background-color: var(--bg-secondary); padding: 15px; border-radius: 8px; margin-top: 20px; text-align: center;">
        <h4 style="margin-top: 0; text-align: left;">Algorithm Output Visualised</h4>
        <img src="{{img_url}}" style="max-width: 100%; border-radius: 8px; border: 1px solid var(--border-color);" alt="LinearRegression.png">
    </div>
    '''
    return render_template(
        'educational_concept.html',
        title="Linear Regression",
        subtitle="The fundamental concept of modeling linear relationships between features and targets.",
        icon="ph ph-chart-line-up",
        content=content,
        math_content=math_content,
        application_content=application_content
    )

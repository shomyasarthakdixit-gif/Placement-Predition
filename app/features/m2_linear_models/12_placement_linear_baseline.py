from flask import render_template
from . import m2_bp

@m2_bp.route('/linear_baseline')
def placement_linear_baseline_page():
    content = r"""
    <p>A Baseline Model provides a starting point for comparing more advanced algorithms. In our project, we have established multiple linear models (Logistic, Ridge, Lasso, ElasticNet, and Softmax) as our baselines to evaluate placement prediction performance.</p>
    <p>Before advancing to complex Tree-based models (like Random Forest or XGBoost), it is crucial to establish the baseline metrics so we can mathematically prove whether the added complexity of advanced models is actually worth it.</p>
    """
    
    math_content = r"""
    <p><strong>Baseline Performance Metrics</strong></p>
    <p>We evaluate our baseline linear models using the standard classification metrics derived from the Confusion Matrix:</p>
    <ul>
        <li><strong>Accuracy:</strong> \\( \\frac{TP + TN}{TP + TN + FP + FN} \\) (Overall correctness)</li>
        <li><strong>Precision:</strong> \\( \\frac{TP}{TP + FP} \\) (When we predict 'Placed', how often are we right?)</li>
        <li><strong>Recall:</strong> \\( \\frac{TP}{TP + FN} \\) (Of all actually placed students, how many did we find?)</li>
        <li><strong>F1 Score:</strong> \\( 2 \\times \\frac{Precision \\times Recall}{Precision + Recall} \\) (Harmonic mean of Precision and Recall)</li>
    </ul>
    """
    
    application_content = r"""
    <p>In this project, our baseline models perform very well due to rigorous feature engineering. The <strong>Logistic Regression (Ridge L2 Penalty)</strong> model achieves roughly ~92% accuracy on the test set.</p>
    <p>However, linear models assume a linear relationship between features and log-odds. Because factors like Aptitude Test Scores and College Tier might interact non-linearly, we proceed to <strong>M3: Tree-Based Models</strong> (Decision Trees, Random Forest, XGBoost) to capture these complex interactions and potentially push our accuracy beyond the linear baseline limits.</p>
    """

    return render_template(
        'educational_concept.html',
        title="Linear Baseline Summary",
        subtitle="Establishing foundational performance metrics before moving to advanced algorithms.",
        icon="ph ph-flag-checkered",
        content=content,
        math_content=math_content,
        application_content=application_content
    )

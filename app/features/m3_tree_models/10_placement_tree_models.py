from flask import render_template
from . import m3_bp

@m3_bp.route('/tree_baseline')
def placement_tree_models_page():
    content = r"""
    <p>Tree-based models represent a massive leap in predictive power compared to linear models because they inherently capture complex, non-linear interactions between variables without requiring explicit feature engineering like polynomial terms.</p>
    <p>For our Student Placement Prediction project, M3 (Tree Models) serves as the core production baseline because of the highly non-linear nature of student performance factors.</p>
    """
    
    math_content = r"""
    <p><strong>Why Trees Outperform Linear Baselines on this Dataset:</strong></p>
    <ul>
        <li><strong>Non-Linear Interactions:</strong> A high CGPA combined with a top-tier college is exponentially better than the sum of its parts. Trees split on these interacting conditions natively.</li>
        <li><strong>Robustness to Outliers:</strong> Unlike logistic regression, trees are completely scale-invariant and immune to extreme numeric outliers.</li>
        <li><strong>Categorical Data:</strong> Advanced gradient boosting algorithms like LightGBM can process categorical variables (like City or Specialization) directly using histogram binning without massive One-Hot Encodings.</li>
    </ul>
    """
    
    application_content = r"""
    <p>In our implementation, we tested several advanced ensemble models:</p>
    <ul>
        <li><strong>Random Forest:</strong> Extremely stable, low variance, excellent baseline.</li>
        <li><strong>AdaBoost:</strong> Iteratively focuses on hard-to-predict students, pushing accuracy higher.</li>
        <li><strong>XGBoost & LightGBM:</strong> The absolute state-of-the-art. LightGBM, in particular, trains exceptionally fast on our 50k rows due to its leaf-wise growth and histogram-based splits.</li>
    </ul>
    <p>The <strong>Random Forest (Classification)</strong> achieves ~95% accuracy, heavily outperforming our M2 Linear baseline of 92%, proving that non-linear relationships dictate placement success.</p>
    """

    return render_template(
        'educational_concept.html',
        title="Tree Models Summary",
        subtitle="Comparing ensemble models and why they form our production baseline.",
        icon="ph ph-tree",
        content=content,
        math_content=math_content,
        application_content=application_content
    )

from flask import render_template
from . import m3_bp

@m3_bp.route('/tree_pruning')
def tree_pruning_page():
    content = r"""
    <p>Decision Trees have a strong tendency to overfit data. If left unconstrained, a tree will continue to split nodes until it perfectly classifies every single training example, resulting in a highly complex tree that performs poorly on unseen test data.</p>
    <p><strong>Pruning</strong> is the technique used to reduce the size of decision trees by removing sections of the tree that provide little power to classify instances.</p>
    """
    
    math_content = r"""
    <p><strong>1. Pre-Pruning (Early Stopping)</strong></p>
    <p>We halt tree construction early before it perfectly classifies the training set. In Scikit-Learn, this is controlled via hyperparameters like:</p>
    <ul>
        <li><code>max_depth</code>: Limits how deep the tree can grow.</li>
        <li><code>min_samples_split</code>: The minimum number of samples required to split an internal node.</li>
        <li><code>min_samples_leaf</code>: The minimum number of samples required to be at a leaf node.</li>
    </ul>
    <hr style="border: 0.5px solid #334155; margin: 20px 0;">
    
    <p><strong>2. Post-Pruning (Cost-Complexity Pruning)</strong></p>
    <p>We allow the tree to grow fully, then trim it back. The Cost-Complexity criterion is minimized:</p>
    <p style="text-align:center;">\\[ R_\\alpha(T) = R(T) + \\alpha |T| \\]</p>
    <p>Where \\( R(T) \\) is the error of tree \\( T \\), \\( |T| \\) is the number of terminal nodes (leaves), and \\( \\alpha \\) is a complexity parameter penalizing large trees.</p>
    """
    
    application_content = r"""
    <p>In <code>ml_pipeline.py</code>, we utilize Pre-Pruning for our Tree-based models to ensure they generalize well to unseen student profiles.</p>
    <p>For example, in our <code>RandomForestClassifier</code>, we explicitly set <code>max_depth=12</code>. Without this constraint, the forest would build incredibly deep trees trying to memorize the 50,000 student rows, which would lead to severe overfitting and slower prediction times.</p>
    """

    return render_template(
        'educational_concept.html',
        title="Tree Pruning",
        subtitle="Preventing Decision Trees from memorizing noise in the training data.",
        icon="ph ph-scissors",
        content=content,
        math_content=math_content,
        application_content=application_content
    )

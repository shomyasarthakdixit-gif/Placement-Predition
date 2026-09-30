from flask import render_template
from . import m2_bp

@m2_bp.route('/gradient_descent')
def gradient_descent_page():
    content = r"""
    <p>Gradient Descent is an iterative optimization algorithm used to find the minimum of a function. In machine learning, it is used to minimize the cost function (such as Mean Squared Error or Log Loss) to find the optimal parameters (weights) for a model.</p>
    <p>By calculating the gradient (slope) of the cost function at the current position, the algorithm takes steps proportional to the negative of the gradient to descend towards the global minimum.</p>
    """
    
    math_content = r"""
    <p>The weight update rule for Gradient Descent is:</p>
    <p style="text-align:center; font-size: 1.3rem;">\\[ \\theta_j := \\theta_j - \\alpha \\frac{\\partial}{\\partial \\theta_j} J(\\theta) \\]</p>
    <p>Where:</p>
    <ul>
        <li>\\( \\theta_j \\) is the weight for the \\( j \\)-th feature.</li>
        <li>\\( \\alpha \\) is the learning rate (step size).</li>
        <li>\\( J(\\theta) \\) is the cost function (e.g., MSE).</li>
        <li>\\( \\frac{\\partial}{\\partial \\theta_j} J(\\theta) \\) is the partial derivative of the cost function with respect to the weight.</li>
    </ul>
    """
    
    application_content = r"""
    <p>In this project, our models (like Logistic Regression with L1/L2 penalties) rely on advanced variations of Gradient Descent. Since our dataset has 50,000 rows, computing the gradient over the entire dataset (Batch Gradient Descent) at every step can be slow.</p>
    <p>Therefore, solvers like <strong>SAGA</strong> (Stochastic Average Gradient Amélioré) or <strong>L-BFGS</strong> are used under the hood by Scikit-Learn to optimize the weights much faster than traditional gradient descent.</p>
    """

    return render_template(
        'educational_concept.html',
        title="Gradient Descent",
        subtitle="The iterative engine powering modern machine learning optimization.",
        icon="ph ph-chart-line-down",
        content=content,
        math_content=math_content,
        application_content=application_content
    )

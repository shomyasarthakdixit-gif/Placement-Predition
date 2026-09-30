from flask import render_template
from . import m2_bp

@m2_bp.route('/normal_equation')
def normal_equation_page():
    content = r"""
    <p>The Normal Equation provides an analytical (closed-form) solution to finding the optimal weights (parameters) for a linear regression model, bypassing the need for iterative optimization algorithms like Gradient Descent.</p>
    <p>By using matrix calculus, we can solve for the optimal weights in a single mathematical step, minimizing the Mean Squared Error (MSE) directly.</p>
    """
    
    math_content = r"""
    <p>To find the optimal weight vector \\( \\theta \\) that minimizes the cost function, we solve the following matrix equation:</p>
    <p style="text-align:center; font-size: 1.3rem;">\\[ \\hat{\\theta} = (X^T X)^{-1} X^T y \\]</p>
    <p>Where:</p>
    <ul>
        <li>\\( \\hat{\\theta} \\) is the vector of optimal weights.</li>
        <li>\\( X \\) is the feature matrix (with a column of 1s added for the bias term).</li>
        <li>\\( X^T \\) is the transpose of matrix \\( X \\).</li>
        <li>\\( (X^T X)^{-1} \\) is the inverse of the matrix product.</li>
        <li>\\( y \\) is the vector of target values.</li>
    </ul>
    """
    
    application_content = r"""
    <p>While the Normal Equation is elegant, calculating the inverse of \\( (X^T X) \\) has a computational complexity of roughly \\( O(n^3) \\), where \\( n \\) is the number of features. Our dataset has multiple encoded features (around 100+ after One-Hot Encoding).</p>
    <p>Because matrix inversion is slow for large feature sets and can fail if the matrix is singular (non-invertible), our Scikit-Learn backend primarily uses advanced iterative solvers (like SAGA or LBFGS) rather than pure Normal Equations for the production-grade Logistic and Ridge models.</p>
    """

    return render_template(
        'educational_concept.html',
        title="Normal Equation",
        subtitle="The analytical approach to finding optimal weights using matrix algebra.",
        icon="ph ph-math-operations",
        content=content,
        math_content=math_content,
        application_content=application_content
    )

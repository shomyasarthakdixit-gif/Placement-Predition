from flask import render_template
from . import m4_bp

@m4_bp.route('/silhouette_score')
def silhouette_score_page():
    content = r"""
    <p>While the Elbow Method looks at the absolute distances within clusters, the <strong>Silhouette Score</strong> measures how similar an object is to its own cluster compared to other clusters.</p>
    <p>It provides a robust mathematical metric (ranging from -1 to 1) to evaluate the density and separation of clusters, ensuring that the chosen \( K \) isn't just mathematically convenient, but actually creates distinct, well-separated groups.</p>
    """
    
    math_content = r"""
    <p>For a single data point \\( i \\):</p>
    <ul>
        <li>Let \\( a(i) \\) be the average distance between \\( i \\) and all other points within the same cluster.</li>
        <li>Let \\( b(i) \\) be the average distance between \\( i \\) and all points in the nearest neighboring cluster.</li>
    </ul>
    <p style="text-align:center; font-size: 1.2rem;">\\[ s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))} \\]</p>
    <p><strong>Interpretation:</strong></p>
    <ul>
        <li><strong>Near +1:</strong> The point is far away from the neighboring clusters (Good).</li>
        <li><strong>Near 0:</strong> The point is on or very close to the decision boundary between two neighboring clusters.</li>
        <li><strong>Near -1:</strong> The point might have been assigned to the wrong cluster.</li>
    </ul>
    """
    
    application_content = r"""
    <p>Calculating the Silhouette Score requires pairwise distances between all points. On our 50,000 row dataset, computing an \( O(N^2) \) metric requires massive memory and compute time.</p>
    <p>During our offline evaluation on a 10% sample, the Silhouette Score peaked at \\( K=4 \\), independently verifying the conclusion drawn by the Elbow Method. As a result, our production K-Means model is hardcoded to 4 clusters.</p>
    """

    return render_template(
        'educational_concept.html',
        title="Silhouette Score",
        subtitle="Evaluating cluster separation and density.",
        icon="ph ph-chart-scatter",
        content=content,
        math_content=math_content,
        application_content=application_content
    )

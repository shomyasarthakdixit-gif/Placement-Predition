from flask import render_template
from . import m4_bp

@m4_bp.route('/hierarchical_clustering')
def hierarchical_clustering_page():
    content = r"""
    <p>Hierarchical Clustering (Agglomerative) builds a hierarchy of clusters using a "bottom-up" approach. Each observation starts in its own cluster, and pairs of clusters are merged as one moves up the hierarchy.</p>
    <p>Unlike K-Means, you do not need to specify the number of clusters in advance, and the algorithm is deterministic (it will always yield the same result for the same data).</p>
    """
    
    math_content = r"""
    <p><strong>Distance and Linkage</strong></p>
    <p>The algorithm relies on two mathematical definitions:</p>
    <ol>
        <li><strong>Distance Metric:</strong> Usually Euclidean distance \\( d(p, q) = \\sqrt{\\sum_{i=1}^n (q_i - p_i)^2} \\).</li>
        <li><strong>Linkage Criterion:</strong> Determines the distance between sets of observations as a function of the pairwise distances between observations. (e.g., Ward's linkage minimizes the variance of clusters being merged).</li>
    </ol>
    """
    
    application_content = r"""
    <p>Hierarchical Clustering has a time complexity of \\( O(N^3) \\) and a space complexity of \\( O(N^2) \\), making it completely unviable for our 50,000-row student dataset.</p>
    <p>Attempting to compute the distance matrix for 50k rows would require roughly 10 GB of RAM just to store the floating-point distances! Therefore, we strictly rely on Mini-Batch K-Means for production, which scales linearly \\( O(N) \\).</p>
    """

    return render_template(
        'educational_concept.html',
        title="Hierarchical Clustering",
        subtitle="Agglomerative approach to grouping data.",
        icon="ph ph-tree-structure",
        content=content,
        math_content=math_content,
        application_content=application_content
    )

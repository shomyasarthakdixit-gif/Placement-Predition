from flask import render_template
from . import m4_bp

@m4_bp.route('/dendrogram')
def dendrogram_page():
    content = r"""
    <p>A <strong>Dendrogram</strong> is a tree-like diagram that records the sequences of merges or splits in Hierarchical Clustering.</p>
    <p>It provides a powerful visual representation of the data's clustering hierarchy, allowing data scientists to "cut" the tree at any specific height to achieve the desired number of clusters.</p>
    """
    
    math_content = r"""
    <p><strong>How to Read a Dendrogram:</strong></p>
    <ul>
        <li><strong>X-axis (Leaves):</strong> Each leaf at the bottom represents an individual data point (or a small cluster).</li>
        <li><strong>Y-axis (Height):</strong> The vertical height of a horizontal connection line represents the distance (or dissimilarity) between the two clusters being merged.</li>
        <li><strong>Cutting the Tree:</strong> Drawing a horizontal line across the dendrogram at a specific height \\( h \\) will intersect several vertical lines. The number of intersected vertical lines equals the number of resulting clusters.</li>
    </ul>
    """
    
    application_content = r"""
    <p>While Dendrograms are excellent for exploratory analysis on small datasets (e.g., \\( N < 1000 \\)), attempting to plot a dendrogram with 50,000 leaves for our student dataset would result in a massive, unreadable black blob at the bottom of the plot.</p>
    <p>For large-scale clustering like ours, the Elbow Method or Silhouette Score combined with K-Means is the industry-standard approach instead of visual dendrogram cuts.</p>
    """

    return render_template(
        'educational_concept.html',
        title="Dendrograms",
        subtitle="Visualizing hierarchical merges.",
        icon="ph ph-git-branch",
        content=content,
        math_content=math_content,
        application_content=application_content
    )

from flask import render_template
from . import m4_bp

@m4_bp.route('/linkages')
def linkages_page():
    content = r"""
    <p>In Hierarchical Clustering, the <strong>Linkage Criterion</strong> determines how the distance between two clusters is calculated when deciding which clusters to merge.</p>
    <p>Choosing the right linkage strongly affects the final cluster shapes. Some linkages tend to produce long, chain-like clusters, while others prefer tight, compact, spherical clusters.</p>
    """
    
    math_content = r"""
    <p><strong>Common Linkage Types:</strong></p>
    <ul>
        <li><strong>Single Linkage (Min):</strong> Distance between two clusters is the distance between their closest points. Tends to produce elongated "chained" clusters.</li>
        <li><strong>Complete Linkage (Max):</strong> Distance between two clusters is the distance between their furthest points. Tends to produce compact clusters.</li>
        <li><strong>Average Linkage:</strong> Distance is the average of all distances between points in the two clusters.</li>
        <li><strong>Ward's Linkage:</strong> Merges the two clusters that result in the smallest increase in the total within-cluster variance (Sum of Squared Errors). This is heavily preferred when looking for spherical clusters, similar to K-Means.</li>
    </ul>
    """
    
    application_content = r"""
    <p>If we were to sample our student dataset and run Agglomerative Clustering, we would use <strong>Ward's Linkage</strong>. Because we used Standard and Robust scalers in our pipeline, our numerical features like CGPA and Aptitude Scores are relatively spherical, which aligns perfectly with Ward's variance-minimizing mathematical approach.</p>
    """

    return render_template(
        'educational_concept.html',
        title="Linkage Criteria",
        subtitle="Determining how to merge clusters in Agglomerative Clustering.",
        icon="ph ph-git-merge",
        content=content,
        math_content=math_content,
        application_content=application_content
    )

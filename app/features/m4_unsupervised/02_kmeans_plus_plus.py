from flask import render_template
from . import m4_bp

@m4_bp.route('/kmeans_plus_plus')
def kmeans_plus_plus_page():
    content = r"""
    <p>Standard K-Means initializes its cluster centers completely at random. If these initial centers are too close to each other, the algorithm can converge to a poor local minimum and take much longer to finish.</p>
    <p><strong>K-Means++</strong> is a smart initialization technique that spreads out the initial cluster centers, ensuring a faster and much more reliable convergence.</p>
    """
    
    math_content = r"""
    <p><strong>The K-Means++ Algorithm:</strong></p>
    <ol>
        <li>Choose one center uniformly at random from the data points.</li>
        <li>For each data point \\( x \\), compute \\( D(x) \\), the distance between \\( x \\) and the nearest center that has already been chosen.</li>
        <li>Choose one new data point at random as a new center, using a weighted probability distribution where a point \\( x \\) is chosen with probability proportional to \\( D(x)^2 \\).</li>
        <li>Repeat Steps 2 and 3 until \\( k \\) centers have been chosen.</li>
        <li>Proceed with standard K-Means optimization.</li>
    </ol>
    """
    
    application_content = r"""
    <p>In Scikit-Learn, K-Means uses K-Means++ initialization by default (<code>init='k-means++'</code>). In our project, all K-Means and Mini-Batch K-Means models leverage this initialization to efficiently cluster the 50,000 student records into the 4 target groups.</p>
    """

    return render_template(
        'educational_concept.html',
        title="K-Means++ Initialization",
        subtitle="Smart centroid placement for faster, more reliable clustering.",
        icon="ph ph-target",
        content=content,
        math_content=math_content,
        application_content=application_content
    )

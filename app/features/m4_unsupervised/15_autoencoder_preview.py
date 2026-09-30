from flask import render_template
from . import m4_bp

@m4_bp.route('/autoencoder_preview')
def autoencoder_preview_page():
    content = r"""
    <p>While PCA (Principal Component Analysis) is our primary tool for dimensionality reduction, it is restricted to finding linear correlations between features.</p>
    <p>An <strong>Autoencoder</strong> is a type of Artificial Neural Network used to learn efficient data codings in an unsupervised manner. It can learn highly complex, non-linear representations of the data.</p>
    """
    
    math_content = r"""
    <p><strong>Architecture of an Autoencoder:</strong></p>
    <ul>
        <li><strong>Encoder (\\( \\phi \\)):</strong> Compresses the high-dimensional input \\( X \\) into a low-dimensional latent space representation (the bottleneck).</li>
        <li><strong>Decoder (\\( \\psi \\)):</strong> Attempts to reconstruct the original input \\( X' \\) from the latent space.</li>
    </ul>
    <p style="text-align:center; font-size: 1.2rem;">\\[ X' = \\psi(\\phi(X)) \\]</p>
    <p>The network is trained using backpropagation to minimize the Reconstruction Loss (e.g., Mean Squared Error between \\( X \\) and \\( X' \\)).</p>
    """
    
    application_content = r"""
    <p>In the context of our Placement Prediction project, an Autoencoder could theoretically be used to reduce the 100+ encoded feature dimensions down to a dense 16-dimension vector before feeding it to our predictive models.</p>
    <p>However, since this is currently a classical Machine Learning syllabus (M1-M4), neural networks (like Autoencoders) are reserved for later advanced Deep Learning modules (M5). Thus, we rely on PCA and t-SNE for dimension reduction.</p>
    """

    return render_template(
        'educational_concept.html',
        title="Autoencoders (Preview)",
        subtitle="Non-linear dimensionality reduction using neural networks.",
        icon="ph ph-brain",
        content=content,
        math_content=math_content,
        application_content=application_content
    )

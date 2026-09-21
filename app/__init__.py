import os
from flask import Flask

def create_app():
    app = Flask(__name__)
                
    # Load ML pipeline
    import joblib
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Output", "models")
    pipeline_file = os.path.join(models_dir, "ml_pipeline_core.pkl")
    
    if os.path.exists(pipeline_file):
        app.config['ML_PIPELINE'] = joblib.load(pipeline_file)
    else:
        print("[WARNING] ml_pipeline_core.pkl not found! Falling back to inline training.")
        from app.core.ml_pipeline import load_data_and_train
        app.config['ML_PIPELINE'] = load_data_and_train(app.root_path)

    # Register Blueprints from the new modular structure
    from app.features.m1_lifecycle import m1_bp
    from app.features.m2_linear_models import m2_bp
    from app.features.m3_tree_models import m3_bp
    from app.features.m4_unsupervised import m4_bp

    app.register_blueprint(m1_bp)
    app.register_blueprint(m2_bp)
    app.register_blueprint(m3_bp)
    app.register_blueprint(m4_bp)

    return app

import os
from flask import Flask

def create_app():
    app = Flask(__name__)
                
    # Load ML pipeline (trains models at startup)
    from app.core.ml_pipeline import load_data_and_train
    app.config['ML_PIPELINE'] = load_data_and_train(app.root_path)
    # Load Salary Regression artifacts
    import joblib
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Output", "models")
    if os.path.exists(models_dir):
        app.config['ADVANCED_MODEL'] = joblib.load(os.path.join(models_dir, 'advanced_regression.pkl'))
        app.config['ADVANCED_PREP'] = joblib.load(os.path.join(models_dir, 'advanced_regression_preprocessor.pkl'))

    # Register Blueprints
    from app.features.home.routes import home_bp
    from app.features.eda.routes import eda_bp
    from app.features.preprocessing.routes import preprocessing_bp
    from app.features.predict.routes import predict_bp
    from app.features.regression.routes import regression_bp

    app.register_blueprint(home_bp)
    app.register_blueprint(eda_bp)
    app.register_blueprint(preprocessing_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(regression_bp)

    return app

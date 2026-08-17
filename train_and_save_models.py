import os
import joblib

def main():
    print("[INIT] Starting Offline Model Training...")
    
    # Needs absolute path for app.root_path logic
    root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
    
    from app.core.ml_pipeline import load_data_and_train
    ml_pipeline = load_data_and_train(root_path)
    
    # Save the entire ml_pipeline dict to disk
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output", "models")
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = os.path.join(out_dir, "ml_pipeline_core.pkl")
    print(f"[INIT] Saving models to {out_file}...")
    joblib.dump(ml_pipeline, out_file, compress=3)
    
    # To save space and memory, we can omit the raw dataframe `df` from the saved dict, 
    # but the frontend uses `df` for unique values in the dropdowns. 
    # Let's save a lighter version of the dataframe if size is an issue, 
    # but gzip compress=3 will make it very small anyway.
    
    print("[INIT] Training and Serialization Complete! You can now start the server safely.")

if __name__ == "__main__":
    main()

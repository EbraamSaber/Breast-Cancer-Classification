
from pathlib import Path
import joblib

from ml_pipeline import run_pipeline, verify_against_notebook

print("Starting model training and evaluation...")

art = run_pipeline()

print("Verifying results against the notebook...")
art["verification"] = verify_against_notebook(art)

output_path = Path(__file__).parent / "model_artifacts.joblib"

print("Saving trained models and results...")
joblib.dump(art, output_path, compress=3)

size_mb = output_path.stat().st_size / (1024 * 1024)

print(f"\nSaved successfully: {output_path}")
print(f"File size: {size_mb:.2f} MB")
print("You can now use the saved artifacts in Streamlit.")

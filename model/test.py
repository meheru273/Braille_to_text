# --- STEP 1: Mount Google Drive ---
from google.colab import drive
import os
import sys

print("Mounting Google Drive...")
drive.mount('/content/drive')
print("Google Drive mounted.")

# --- STEP 2: Configure Paths ---
# IMPORTANT: Update 'GOOGLE_DRIVE_PROJECT_PATH' if your 'fcos_scratch' folder is NOT directly in 'MyDrive'
GOOGLE_DRIVE_PROJECT_PATH = 'https://drive.google.com/drive/folders/1CWHIradU9dI3wsN6cZf9Rcw5XAbmC0Vy?usp=drive_link' # <-- CHECK THIS PATH

# Check if the project folder exists in Drive
if not os.path.exists(GOOGLE_DRIVE_PROJECT_PATH):
    raise FileNotFoundError(
        f"Project folder not found at {GOOGLE_DRIVE_PROJECT_PATH}. "
        f"Please check the path. Ensure the 'fcos_scratch' folder (with runs/, dsbi.coco/, custom.coco/, and .py files) "
        f"is uploaded to this location in your Google Drive."
    )

# Add the project directory to Python path so imports like 'from Dataset import ...' work
sys.path.append(GOOGLE_DRIVE_PROJECT_PATH)

# Change the working directory to the project folder
# This is crucial for relative paths (like loading models from 'runs/') within your script to work correctly
os.chdir(GOOGLE_DRIVE_PROJECT_PATH)
print(f"Working directory changed to: {os.getcwd()}")

# --- STEP 3: Import and Run Evaluation ---
# Import your evaluation script (assuming it's named evaluate.py and in the project root)
try:
    import model.evaluation as evaluation
    print("Successfully imported traininscript.py")
except ImportError as e:
    print(f"Error importing traininscript.py: {e}")
    print("Please make sure:")
    print("1. evaluate.py is uploaded to your Google Drive project folder.")
    print("2. The path GOOGLE_DRIVE_PROJECT_PATH is set correctly.")
    print("3. All required .py files (Dataset.py, FPN.py, etc.) are also in the folder.")
    raise

# --- STEP 4: Modify Paths in the Script (Dynamically) ---
# We override the hardcoded paths in the original script with paths relative to the mounted Drive folder
def patched_main():
    """Patched main function using paths from Google Drive"""
    import torch
    from model.evaluation import evaluate_model_on_dataset, MetricsCalculator # Import necessary functions/classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use the current working directory (which is set to the project folder in Drive)
    base_path = os.getcwd()
    print(f"Base path set to (from Google Drive): {base_path}")

    runs_path = os.path.join(base_path, "runs")
    # Model paths - relative to base_path (now pointing to Drive)
    models = {
        'baseline_dsbi': os.path.join(runs_path, 'baseline_all_false_dsbi.pth'),
        'cbam_only_dsbi': os.path.join(runs_path, 'cbam_only_dsbi.pth'),
        'coord_cbam_dsbi': os.path.join(runs_path, 'coord_cbam_dsbi.pth'),
    }

    # Dataset paths - relative to base_path (now pointing to Drive)
    datasets = {
        'dsbi': os.path.join(base_path, 'dsbi.coco', 'valid'),
        'custom': os.path.join(base_path, 'custom.coco', 'valid')
    }

    # Check available models and datasets
    available_models = {k: v for k, v in models.items() if os.path.exists(v)}
    available_datasets = {k: v for k, v in datasets.items() if os.path.exists(v)}
    print(f"Found {len(available_models)} models, {len(available_datasets)} datasets")

    if not available_models or not available_datasets:
        print("No models or datasets found! Check file paths in Google Drive.")
        return

    # Results storage
    all_results = {}
    # Evaluate models
    for model_name, model_path in available_models.items():
        model_results = {}
        # Determine which datasets to test (logic from original script)
        if 'dsbi' in model_name and 'dsbi' in available_datasets:
            datasets_to_test = {'dsbi': available_datasets['dsbi']}
        elif 'custom' in model_name and 'custom' in available_datasets:
            datasets_to_test = {'custom': available_datasets['custom']}
        else:
            datasets_to_test = available_datasets

        for dataset_name, dataset_path in datasets_to_test.items():
            print(f"\n{'='*60}")
            print(f"Testing {model_name} on {dataset_name} validation set")
            print(f"{'='*60}")
            try:
                # Call the core evaluation function from your script
                results = evaluate_model_on_dataset(
                    model_path, dataset_path, device
                )
                model_results[dataset_name] = results
                print(f"\n✓ Results for {model_name} on {dataset_name}:")
                print(f"  Precision: {results['precision']:.4f}")
                print(f"  Recall: {results['recall']:.4f}")
                print(f"  F1-Score: {results['f1']:.4f}")
                print(f"  mAP@50: {results['mAP@50']:.4f}")
                print(f"  mAP@70: {results['mAP@70']:.4f}")
                print(f"  Predictions: {results['num_predictions']}")
                print(f"  Ground Truth: {results['num_ground_truth']}")
            except Exception as e:
                print(f"✗ Error evaluating {model_name} on {dataset_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        if model_results:
            all_results[model_name] = model_results

    # --- STEP 5: Create Summary and Save Results ---
    if all_results:
        import pandas as pd
        import json
        summary_data = []
        for model_name, model_results in all_results.items():
            for dataset_name, results in model_results.items():
                summary_data.append({
                    'Model': model_name,
                    'Dataset': dataset_name,
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1-Score': results['f1'],
                    'mAP@50': results['mAP@50'],
                    'mAP@70': results['mAP@70'],
                    'Num_Predictions': results['num_predictions'],
                    'Num_Ground_Truth': results['num_ground_truth']
                })
        df = pd.DataFrame(summary_data)

        # Save results to the project folder in Drive
        output_dir = os.path.join(base_path, 'evaluation_results_colab')
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'model_evaluation_results.csv')
        df.to_csv(csv_path, index=False)

        print(f"\n{'='*120}")
        print("VALIDATION SET EVALUATION SUMMARY (Run in Colab)")
        print("="*120)
        print(df.to_string(index=False, float_format='%.4f'))

        # Save detailed results
        json_path = os.path.join(output_dir, 'detailed_results.json')
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n✓ Results saved to Google Drive: {csv_path}")
        print(f"✓ Detailed results saved to Google Drive: {json_path}")
    else:
        print("\nNo results to save.")

# --- STEP 6: Execute the Patched Evaluation ---
print("\nStarting evaluation using data from Google Drive...")
patched_main()
print("\nEvaluation script finished.")

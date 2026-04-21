# main.py
# Person 5: Main pipeline orchestrator
from train import train_pipeline
from inference import generate_answer

def run_project():
    print("========================================")
    print(" Starting QA Transformer Pipeline")
    print("========================================")
    
    print("\n[Step 1] Initiate Training")
    try:
        train_pipeline()
    except Exception as e:
        print(f"Training encountered an error: {e}")
        
    print("\n[Step 2] Testing Inference")
    try:
        sample_q = "What is the best pesticide for maize?"
        generate_answer(sample_q)
    except Exception as e:
        print(f"Inference encountered an error: {e}")
        
    print("\n========================================")
    print(" Pipeline execution finished.")
    print("========================================")

if __name__ == "__main__":
    run_project()

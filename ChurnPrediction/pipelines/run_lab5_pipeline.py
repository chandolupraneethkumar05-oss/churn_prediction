import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def execute_pipeline():
    print("[INFO] =========================================")
    print("[INFO] Starting Lab 5: Production Data Pipeline")
    print("[INFO] =========================================")

    scripts = [
        "pipelines/preprocess_pipeline.py",
        "outputs/validate_outputs.py"
    ]

    for script in scripts:
        print(f"\n[INFO] ---> Executing {script}...")
        result = subprocess.run([sys.executable, script], cwd=PROJECT_ROOT)

        if result.returncode != 0:
            print(f"[ERROR] Pipeline halted. {script} caught an error.")
            sys.exit(1)

    print("\n[SUCCESS] Lab 5 Production Pipeline fully executed!")

if __name__ == "__main__":
    execute_pipeline()

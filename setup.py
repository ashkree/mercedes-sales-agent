# setup.py

import os
import subprocess
import shutil
from pathlib import Path


def run_command(command, desc=None):
    """Run a command and print its description"""
    if desc:
        print(f"⏳ {desc}...")

    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        return False


def main():
    # Get project root
    project_root = Path(__file__).resolve().parent
    model_path = project_root / "models" / "phi-2.Q4_K_M.gguf"
    csv_path = project_root / "data" / "raw" / "gargash_mercedes_models.csv"
    csv2chunks_path = project_root / "scripts" / "csv2chunks.py"

    # Check dependencies
    print("Checking dependencies...")

    if not model_path.exists():
        print(f"❌ Error: Phi-2 model not found at {model_path}")
        print("Please ensure the model file is placed in the models directory.")
        return False
    else:
        print(f"✅ Found Phi-2 model at {model_path}")

    if not csv_path.exists():
        print(f"❌ Error: CSV file not found at {csv_path}")
        print("Please ensure the CSV file is in the data/raw directory.")
        return False
    else:
        print(f"✅ Found Mercedes data CSV at {csv_path}")

    if not csv2chunks_path.exists():
        print(f"❌ Error: csv2chunks.py not found at {csv2chunks_path}")
        print("Please ensure the script is in the scripts directory.")
        return False
    else:
        print(f"✅ Found csv2chunks.py script at {csv2chunks_path}")

    # Create required directories
    for directory in ["data", "data/raw", "data/faiss"]:
        os.makedirs(os.path.join(project_root, directory), exist_ok=True)
        print(f"✅ Created directory: {directory}")

    # Run the CSV to chunks script
    print("⏳ Converting CSV to chunks...")
    try:
        subprocess.run(["python", csv2chunks_path],
                       cwd=project_root, check=True)

        # Move docstore.jsonl to the correct location if it's in the project root
        docstore_source = project_root / "docstore.jsonl"
        docstore_target = project_root / "data" / "raw" / "docstore.jsonl"

        if docstore_source.exists():
            shutil.copy(docstore_source, docstore_target)
            print(f"✅ Copied docstore.jsonl to {docstore_target}")
        elif not docstore_target.exists():
            print("❌ Error: docstore.jsonl was not created")
            return False
        else:
            print(f"✅ Found docstore.jsonl at {docstore_target}")
    except Exception as e:
        print(f"❌ Error running csv2chunks.py: {e}")
        return False

    # Create vector embeddings
    print("⏳ Creating vector embeddings...")
    try:
        from scripts.embed_chunks import main as embed_main
        embed_main()
        print("✅ Vector embeddings created successfully")
    except Exception as e:
        print(f"❌ Error creating vector embeddings: {e}")
        return False

    # Print setup summary
    print("\n✅ Setup completed successfully! The system is ready to use.")
    print("\nRun the assistant with: python main.py")


if __name__ == "__main__":
    main()

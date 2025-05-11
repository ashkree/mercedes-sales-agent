# scripts/csv2chunks.py

import json
import pandas as pd
from pathlib import Path


def main():
    # Use absolute paths based on the project root
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "data" / "raw" / "gargash_mercedes_models.csv"
    output_path = base_dir / "data" / "raw" / "docstore.jsonl"

    print(f"Loading CSV from: {csv_path}")

    # Load the CSV
    df = pd.read_csv(csv_path)

    # Define a formatter function
    def row_to_chunk(row):
        return {
            "id": f"{row['model_name'].strip().lower().replace(' ', '_')}",
            "text": (
                f"Model: {row['model_name']} | "
                f"Body Style: {row['body_style']} | "
                f"Powertrain: {row['powertrain']} | "
                f"Seats: {row['number_of_seats']} | "
                f"Starting Price: AED {row['starting_price_aed']}"
            )
        }

    # Apply to all rows
    chunks = [row_to_chunk(row) for _, row in df.iterrows()]

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSONL format
    print(f"Saving {len(chunks)} chunks to: {output_path}")
    with open(output_path, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

    print(
        f"✅ Successfully created docstore with {len(chunks)} Mercedes-Benz models")


if __name__ == "__main__":
    main()

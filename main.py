#!/usr/bin/env python3
"""
Main Entry Point - Process Handelsregister Data
"""

import sys
import json
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'src')

from pipeline import build_controller


def main():
    """Process handelsregister CSV and extract keywords."""

    print("="*80)
    print("KEYWORD EXTRACTION - HANDELSREGISTER PROCESSING")
    print("="*80)

    # Configuration
    CSV_FILE = "data/raw/handelsregister_sample_10k.csv"
    OUTPUT_FILE = "output/results.json"
    BATCH_SIZE = 100

    # Check input file
    if not Path(CSV_FILE).exists():
        print(f"✗ {CSV_FILE} not found.")
        print("  The raw corpus is not distributed with this repository; see")
        print("  data/README.md. Everything else still runs: `make reproduce`")
        print("  regenerates the reported results from committed data.")
        return

    print(f"\n[1/5] Loading CSV file: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
    print(f"      Loaded {len(df)} records")
    print(f"      Columns: {list(df.columns)}")

    # Initialize components from config/config.yaml
    print(f"\n[2/5] Initializing components...")
    controller, config = build_controller()
    top_n_keywords = config["extraction"]["top_n_final"]
    print(f"      ✓ Components initialized ({top_n_keywords} keywords per document)")

    # Process data
    print(f"\n[3/5] Processing data ({len(df)} documents)...")

    results = []
    errors = []

    # Process in batches
    for batch_start in range(0, len(df), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(df))
        batch_texts = df.iloc[batch_start:batch_end]['purpose'].tolist()

        print(f"      Processing batch {batch_start//BATCH_SIZE + 1}/{(len(df)-1)//BATCH_SIZE + 1}...", end='', flush=True)

        try:
            batch_results = controller.extract_batch(
                batch_texts,
                top_n_keywords=top_n_keywords,
                use_validation=False,
                show_progress=False
            )

            results.extend(batch_results)
            successful = sum(1 for r in batch_results if r['status'] == 'success')
            print(f" ✓ ({successful}/{len(batch_texts)} successful)")

        except Exception as e:
            print(f" ✗ Error: {e}")
            errors.append({'batch': batch_start, 'error': str(e)})

    # Save results
    print(f"\n[4/5] Saving results to {OUTPUT_FILE}...")
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"      ✓ Saved {len(results)} results")

    # Summary statistics
    print(f"\n[5/5] Summary Statistics")
    print("="*80)

    stats = controller.get_extraction_stats(results)
    print(f"Total documents:          {stats['total_documents']}")
    print(f"Successful:               {stats['successful']} ({stats['success_rate']:.1%})")
    print(f"Failed:                   {stats['failed']}")
    print(f"Total keywords extracted: {stats['total_keywords_extracted']}")
    print(f"Avg keywords/document:    {stats['avg_keywords_per_document']:.1f}")
    print(f"Avg confidence score:     {stats['avg_confidence']:.3f}")

    print(f"\nSector Distribution:")
    for sector, count in sorted(stats['sector_distribution'].items()):
        pct = (count / stats['successful']) * 100
        print(f"  {sector:2s}: {count:5d} ({pct:5.1f}%)")

    print("\n" + "="*80)
    print("✓ Processing Complete!")
    print("="*80)

    if errors:
        print(f"\n⚠ {len(errors)} errors occurred during processing")
        for error in errors[:5]:
            print(f"  - Batch {error['batch']}: {error['error']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run unified preprocessing pipeline

This script loads raw ICU data and generates preprocessed datasets
in all required formats for different models.

Usage:
    python run_preprocessing.py
    
    Or with custom paths:
    python run_preprocessing.py \
        --data-dir Data_group_7 \
        --output-dir preprocessed_data
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from data_processing import UnifiedPreprocessor


def main():
    parser = argparse.ArgumentParser(
        description="Run unified ICU data preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: use Data_group_7
  python run_preprocessing.py

  # Custom data directory
  python run_preprocessing.py --data-dir Data_group_7

  # Custom output directory
  python run_preprocessing.py --output-dir my_preprocessed_data
        """
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(REPO_ROOT / "Data_group_7"),
        help="Path to Data_group_7 directory (default: <repo>/Data_group_7)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "preprocessed_data"),
        help="Output directory for preprocessed data (default: <repo>/preprocessed_data)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Verify data directory exists
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    features_dir = data_dir / "Features_group_7" / "Features_group_10"
    outcomes_file = data_dir / "Outcomes-7.txt"
    
    if not features_dir.exists():
        print(f"❌ Error: Features directory not found: {features_dir}")
        sys.exit(1)
    
    if not outcomes_file.exists():
        print(f"❌ Error: Outcomes file not found: {outcomes_file}")
        sys.exit(1)
    
    print(f"📂 Data directory: {data_dir}")
    print(f"📁 Output directory: {output_dir}\n")
    
    try:
        # Run preprocessing
        preprocessor = UnifiedPreprocessor(data_dir=str(data_dir))
        preprocessor.prepare_all_formats(output_dir=str(output_dir))
        
        print("\n✅ Preprocessing completed successfully!")
        print(f"\n📍 Output saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

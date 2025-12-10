"""
main.py
--------
Entry point for the Portfolio Analyzer.

Coordinates the workflow by:
- Parsing input arguments
- Loading strategy data via data_loader.py
- Calculating metrics via metrics.py
- Building the HTML report via report_html_generator.py

Outputs a complete portfolio analysis report to the reports/ directory.
"""

from datetime import datetime
from pathlib import Path
import sys
import argparse


from config import DATASET_FOLDER
from report_generator import process_all_strategies, process_folder


def main():
    """Main execution function"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze trading strategy performance")
    parser.add_argument(
        "folder",
        nargs="?",
        default=None,
        help="Specific subfolder to analyze (default: all subfolders)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("Portfolio Analyzer - Trading Strategy Combination Tool")
    print("=" * 60 + "\n")

    # Check if dataset folder exists
    dataset_path = Path(DATASET_FOLDER)
    if not dataset_path.exists():
        print(f"❌ ERROR: Dataset folder '{DATASET_FOLDER}' not found!")
        print(f"   Please create the folder and add your CSV files.")
        sys.exit(1)

    # Find all subfolders (ignore files directly in dataset/)
    subfolders = [d for d in dataset_path.iterdir() if d.is_dir()]

    if len(subfolders) == 0:
        print(f"❌ ERROR: No subfolders found in '{DATASET_FOLDER}' folder!")
        print(f"   Please create subfolders like: swing_systems/, day_trades/, etc.")
        sys.exit(1)

    # If specific folder requested, validate it
    if args.folder:
        requested_folder = dataset_path / args.folder
        if not requested_folder.exists() or not requested_folder.is_dir():
            print(f"❌ ERROR: Folder '{args.folder}' not found in '{DATASET_FOLDER}'!")
            print(f"\n   Available folders:")
            for folder in subfolders:
                print(f"     • {folder.name}")
            sys.exit(1)
        folders_to_process = [requested_folder]
    else:
        folders_to_process = subfolders

    print(f"📁 Found {len(subfolders)} strategy group(s) in {DATASET_FOLDER}/\n")
    if args.folder:
        print(f"🎯 Processing only: {args.folder}\n")

    # Single timestamp for all reports
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Process each folder
    results = []
    for folder_path in folders_to_process:
        try:
            result = process_folder(folder_path, folder_path.name)
            if result:
                results.append(result)
        except KeyboardInterrupt:
            print("\n\n⚠️  Analysis interrupted by user.")
            sys.exit(1)
        except Exception as e:
            print(f"  ❌ ERROR processing {folder_path.name}: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    # NEW: Generate combined "all strategies" report (only if not processing a specific folder)
    if not args.folder:
        try:
            all_result = process_all_strategies(dataset_path)
            if all_result:
                results.append(all_result)
        except Exception as e:
            print(f"  ❌ ERROR generating combined report: {str(e)}")
            import traceback

            traceback.print_exc()

    # Final summary
    print("=" * 60)
    if len(results) == 0:
        print("⚠️  No reports generated - all folders were empty or had errors")
    else:
        print(f"✅ ANALYSIS COMPLETE - {len(results)} report(s) generated")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

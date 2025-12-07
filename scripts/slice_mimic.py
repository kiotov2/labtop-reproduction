#!/usr/bin/env python3
"""
Slice MIMIC-IV Dataset for Small-Scale Reproduction
====================================================

Creates a small subset of MIMIC-IV data for quick local experiments.

Usage:
    python scripts/slice_mimic.py --source /path/to/mimic-iv-3.1 --dest ./data_small --n_stays 200

PhysioNet DUA Compliance:
    - This script processes local MIMIC-IV data you have access to
    - Output data remains on your local machine
    - DO NOT upload the sliced data to public repositories
"""

import pandas as pd
import argparse
from pathlib import Path
import sys
from typing import Optional

class MIMICSlicer:
    """Handles slicing of MIMIC-IV dataset to a manageable subset."""

    # Tables that use stay_id for filtering
    ICU_STAY_TABLES = {
        'icu/icustays.csv': {'key': 'stay_id'},
        'icu/inputevents.csv': {'key': 'stay_id'},
        'icu/procedureevents.csv': {'key': 'stay_id'},
        'icu/outputevents.csv': {'key': 'stay_id'},
    }

    # Tables that use hadm_id (hospital admission) for filtering
    HOSP_HADM_TABLES = {
        'hosp/labevents.csv': {'key': 'hadm_id'},
        'hosp/emar.csv': {'key': 'hadm_id'},
        'hosp/emar_detail.csv': {'key': 'emar_id', 'join_via': 'hosp/emar.csv'},
        'hosp/microbiologyevents.csv': {'key': 'hadm_id'},
        'hosp/admissions.csv': {'key': 'hadm_id'},
    }

    # Tables that use subject_id (patient) for filtering
    PATIENT_TABLES = {
        'hosp/patients.csv': {'key': 'subject_id'},
    }

    # Reference/description tables (keep entire table)
    REFERENCE_TABLES = [
        'hosp/d_labitems.csv',
        'icu/d_items.csv',
    ]

    def __init__(self, source_path: Path, dest_path: Path, n_stays: int = 200):
        self.source = Path(source_path)
        self.dest = Path(dest_path)
        self.n_stays = n_stays

        # Will be populated during slicing
        self.selected_stay_ids = None
        self.selected_hadm_ids = None
        self.selected_subject_ids = None

    def _detect_file(self, relative_path: str) -> Optional[Path]:
        """Detect if file exists as .csv or .csv.gz"""
        csv_path = self.source / relative_path
        gz_path = self.source / f"{relative_path}.gz"

        if csv_path.exists():
            return csv_path
        elif gz_path.exists():
            return gz_path
        else:
            return None

    def _read_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Read CSV file, handling both .csv and .csv.gz"""
        if str(file_path).endswith('.gz'):
            return pd.read_csv(file_path, compression='gzip', **kwargs)
        else:
            return pd.read_csv(file_path, **kwargs)

    def _save_csv(self, df: pd.DataFrame, relative_path: str):
        """Save DataFrame to destination as .csv.gz, preserving directory structure"""
        dest_file = self.dest / f"{relative_path}.gz"
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest_file, index=False, compression='gzip')
        print(f"  ✓ Saved {len(df):,} rows -> {relative_path}.gz")

    def step1_select_stays(self):
        """Select random ICU stays and extract related IDs"""
        print(f"\n[1/4] Selecting {self.n_stays} ICU stays...")

        icustays_file = self._detect_file('icu/icustays.csv')
        if not icustays_file:
            raise FileNotFoundError("icu/icustays.csv not found in source")

        # Load and sample ICU stays
        icustays = self._read_csv(icustays_file)

        # Filter by minimum length of stay (optional: keep stays >= 6 hours)
        icustays['los_hours'] = icustays['los'] * 24
        icustays_filtered = icustays[icustays['los_hours'] >= 6].copy()

        if len(icustays_filtered) < self.n_stays:
            print(f"  ⚠ Warning: Only {len(icustays_filtered)} stays >= 6h available")
            sampled = icustays_filtered
        else:
            sampled = icustays_filtered.sample(n=self.n_stays, random_state=42)

        # Extract IDs
        self.selected_stay_ids = set(sampled['stay_id'].unique())
        self.selected_hadm_ids = set(sampled['hadm_id'].dropna().unique())
        self.selected_subject_ids = set(sampled['subject_id'].unique())

        # Save sampled stays
        self._save_csv(sampled, 'icu/icustays.csv')

        print(f"  Selected: {len(self.selected_stay_ids)} stays, "
              f"{len(self.selected_hadm_ids)} admissions, "
              f"{len(self.selected_subject_ids)} patients")

    def step2_filter_icu_tables(self):
        """Filter ICU tables by stay_id"""
        print(f"\n[2/4] Filtering ICU event tables...")

        for relative_path, config in self.ICU_STAY_TABLES.items():
            if relative_path == 'icu/icustays.csv':
                continue  # Already handled

            source_file = self._detect_file(relative_path)
            if not source_file:
                print(f"  ⚠ Skipping {relative_path} (not found)")
                continue

            print(f"  Processing {relative_path}...")
            key = config['key']

            # Read in chunks to handle large files
            chunks = []
            chunk_num = 0
            rows_processed = 0
            rows_kept = 0
            
            for chunk in pd.read_csv(source_file, chunksize=100000, low_memory=False):
                chunk_num += 1
                rows_processed += len(chunk)
                
                if key in chunk.columns:
                    filtered = chunk[chunk[key].isin(self.selected_stay_ids)]
                    if len(filtered) > 0:
                        chunks.append(filtered)
                        rows_kept += len(filtered)
                
                # Print progress every 5 chunks
                if chunk_num % 5 == 0:
                    print(f"    → Chunk {chunk_num}: {rows_processed:,} rows scanned, {rows_kept:,} kept")

            if chunks:
                df = pd.concat(chunks, ignore_index=True)
                self._save_csv(df, relative_path)
            else:
                print(f"  ⚠ No matching rows found for {relative_path}")

    def step3_filter_hospital_tables(self):
        """Filter hospital tables by hadm_id or subject_id"""
        print(f"\n[3/4] Filtering hospital tables...")

        # First, handle emar to get emar_ids for emar_detail
        emar_ids = None

        for relative_path, config in self.HOSP_HADM_TABLES.items():
            source_file = self._detect_file(relative_path)
            if not source_file:
                print(f"  ⚠ Skipping {relative_path} (not found)")
                continue

            print(f"  Processing {relative_path}...")

            # Special handling for emar_detail (filter by emar_id)
            if config.get('join_via'):
                if emar_ids is None:
                    print(f"  ⚠ Cannot filter {relative_path}: emar not processed yet")
                    continue
                key = config['key']
                chunks = []
                chunk_num = 0
                rows_processed = 0
                rows_kept = 0
                
                for chunk in pd.read_csv(source_file, chunksize=100000, low_memory=False):
                    chunk_num += 1
                    rows_processed += len(chunk)
                    
                    if key in chunk.columns:
                        filtered = chunk[chunk[key].isin(emar_ids)]
                        if len(filtered) > 0:
                            chunks.append(filtered)
                            rows_kept += len(filtered)
                    
                    if chunk_num % 5 == 0:
                        print(f"    → Chunk {chunk_num}: {rows_processed:,} rows scanned, {rows_kept:,} kept")
                        
                if chunks:
                    df = pd.concat(chunks, ignore_index=True)
                    self._save_csv(df, relative_path)
                continue

            # Regular filtering
            key = config['key']
            chunks = []
            chunk_num = 0
            rows_processed = 0
            rows_kept = 0
            
            for chunk in pd.read_csv(source_file, chunksize=100000, low_memory=False):
                chunk_num += 1
                rows_processed += len(chunk)
                
                if key in chunk.columns:
                    filtered = chunk[chunk[key].isin(self.selected_hadm_ids)]
                    if len(filtered) > 0:
                        chunks.append(filtered)
                        rows_kept += len(filtered)
                
                # Print progress every 5 chunks (labevents has ~200 chunks, so ~40 updates)
                if chunk_num % 5 == 0:
                    print(f"    → Chunk {chunk_num}: {rows_processed:,} rows scanned, {rows_kept:,} kept")

            if chunks:
                df = pd.concat(chunks, ignore_index=True)

                # Save emar_ids for emar_detail filtering
                if relative_path == 'hosp/emar.csv' and 'emar_id' in df.columns:
                    emar_ids = set(df['emar_id'].unique())

                self._save_csv(df, relative_path)
            else:
                print(f"  ⚠ No matching rows found for {relative_path}")

        # Filter patient tables by subject_id
        for relative_path, config in self.PATIENT_TABLES.items():
            source_file = self._detect_file(relative_path)
            if not source_file:
                print(f"  ⚠ Skipping {relative_path} (not found)")
                continue

            print(f"  Processing {relative_path}...")
            df = self._read_csv(source_file)
            key = config['key']

            if key in df.columns:
                filtered = df[df[key].isin(self.selected_subject_ids)]
                self._save_csv(filtered, relative_path)

    def step4_copy_reference_tables(self):
        """Copy reference/description tables entirely"""
        print(f"\n[4/4] Copying reference tables...")

        for relative_path in self.REFERENCE_TABLES:
            source_file = self._detect_file(relative_path)
            if not source_file:
                print(f"  ⚠ Skipping {relative_path} (not found)")
                continue

            print(f"  Copying {relative_path}...")
            df = self._read_csv(source_file)
            self._save_csv(df, relative_path)

    def execute(self):
        """Run the complete slicing pipeline"""
        print("=" * 60)
        print("MIMIC-IV Dataset Slicer for Small-Scale Reproduction")
        print("=" * 60)
        print(f"Source: {self.source}")
        print(f"Destination: {self.dest}")
        print(f"Target: {self.n_stays} ICU stays")

        # Create destination directory
        self.dest.mkdir(parents=True, exist_ok=True)

        # Execute pipeline
        self.step1_select_stays()
        self.step2_filter_icu_tables()
        self.step3_filter_hospital_tables()
        self.step4_copy_reference_tables()

        print("\n" + "=" * 60)
        print("✓ Slicing complete!")
        print("=" * 60)
        print(f"\nSmall dataset saved to: {self.dest}")
        print(f"\n⚠ IMPORTANT: This data is covered by PhysioNet DUA.")
        print("   Do NOT upload to GitHub or share publicly.")
        print("\nNext steps:")
        print("  1. Update your config to point to this data:")
        print(f"     raw_data_path: {self.dest.absolute()}")
        print("  2. Run preprocessing:")
        print("     cd labtop/src && python scripts/preprocess.py")


def main():
    parser = argparse.ArgumentParser(
        description="Create a small subset of MIMIC-IV for local experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python scripts/slice_mimic.py \\
    --source /home/data_storage/mimic-iv-3.1 \\
    --dest ./data_small \\
    --n_stays 200

This creates a manageable dataset for CPU-based training and testing.
The script outputs compressed .csv.gz files matching MIMIC-IV format.
        """
    )

    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to full MIMIC-IV dataset (e.g., /path/to/mimic-iv-3.1)'
    )

    parser.add_argument(
        '--dest',
        type=str,
        default='./data_small',
        help='Destination path for sliced dataset (default: ./data_small)'
    )

    parser.add_argument(
        '--n_stays',
        type=int,
        default=200,
        help='Number of ICU stays to sample (default: 200)'
    )

    args = parser.parse_args()

    # Validate source path
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source path does not exist: {source_path}")
        sys.exit(1)

    # Execute slicing
    slicer = MIMICSlicer(
        source_path=source_path,
        dest_path=Path(args.dest),
        n_stays=args.n_stays
    )

    try:
        slicer.execute()
    except Exception as e:
        print(f"\n✗ Error during slicing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

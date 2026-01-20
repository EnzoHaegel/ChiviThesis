import pandas as pd
import os
import glob

# Define paths
source_dir = 'dta_raw'
output_dir = 'csv_raw'

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Find all .dta files
dta_files = glob.glob(os.path.join(source_dir, '*.dta'))

if not dta_files:
    print(f"No .dta files found in {source_dir}")
else:
    print(f"Converting {len(dta_files)} files...")

    for dta_file in dta_files:
        try:
            # Read .dta file
            print(f"Reading {dta_file}...")
            df = pd.read_stata(dta_file)

            # Construct output filename
            base_name = os.path.basename(dta_file)
            csv_name = base_name.replace('.dta', '.csv')
            output_path = os.path.join(output_dir, csv_name)

            # Save to CSV
            print(f"Saving to {output_path}...")
            df.to_csv(output_path, index=False)
            print("Done.")

        except Exception as e:
            print(f"Error converting {dta_file}: {e}")

    print("Conversion finished.")

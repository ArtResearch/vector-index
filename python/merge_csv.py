import pandas as pd
import argparse
import os
import sys
import re
import linecache

def merge_csv_files(primary_csv_path, secondary_csv_path, output_csv_path):
    """
    Merges a 'dataset' column from a secondary CSV into a primary CSV based on a common 'uri' column.
    This version loads the entire primary file into memory.

    Args:
        primary_csv_path (str): Path to the main CSV file (the one to be extended).
        secondary_csv_path (str): Path to the CSV file containing the 'uri' and 'dataset' columns.
        output_csv_path (str): Path to save the merged CSV file.
    """
    try:
        # Load the secondary CSV into memory. This file is expected to be smaller
        # and will serve as a lookup table for the 'dataset' information.
        print(f"Loading secondary data from '{secondary_csv_path}'...")
        secondary_df = pd.read_csv(secondary_csv_path, usecols=['uri', 'dataset'])
        secondary_df.set_index('uri', inplace=True)
        print("Secondary data loaded and indexed.")

        print(f"Loading primary data from '{primary_csv_path}'...")
        try:
            primary_df = pd.read_csv(primary_csv_path)
        except pd.errors.ParserError as e:
            print(f"\n--- Parsing Error ---", file=sys.stderr)
            print(f"Pandas could not parse the primary CSV file '{os.path.basename(primary_csv_path)}'.", file=sys.stderr)
            print(f"Error details: {e}", file=sys.stderr)
            
            # Use regex to find the line number from the error message
            match = re.search(r'line (\d+)', str(e))
            if match:
                line_number = int(match.group(1))
                print(f"\nThe error seems to be on or around line {line_number}.", file=sys.stderr)
                print("Here is the content of that line:", file=sys.stderr)
                
                # Use linecache to efficiently get the specific line from the file
                line = linecache.getline(primary_csv_path, line_number)
                if line:
                    print(f"\n>>> LINE {line_number}: {line.strip()}\n", file=sys.stderr)
                    print("Please check this line for issues like extra commas or incorrect quoting.", file=sys.stderr)
                else:
                    print(f"Could not retrieve line {line_number} to display it.", file=sys.stderr)
            
            print("---------------------\n", file=sys.stderr)
            return # Stop execution

        print("Primary data loaded.")

        # Perform a left merge to add the 'dataset' column.
        # The row order of the primary_df is preserved by default.
        print("Merging dataframes...")
        merged_df = pd.merge(primary_df, secondary_df, on='uri', how='left')

        # Save the final merged dataframe to the output file.
        print(f"Saving merged data to '{output_csv_path}'...")
        merged_df.to_csv(output_csv_path, index=False)

        print(f"Successfully merged data and saved to '{output_csv_path}'.")

    except FileNotFoundError as e:
        print(f"Error: The file was not found - {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(
        description="""
        Merge a 'dataset' column from a secondary CSV file into a primary CSV file.
        This script loads the entire primary file into memory for the merge operation.
        The join is performed on a common 'uri' column, and the original row order is preserved.
        """
    )
    parser.add_argument("primary_csv", help="Path to the primary CSV file (the large file to be extended).")
    parser.add_argument("secondary_csv", help="Path to the secondary CSV file (containing 'uri' and 'dataset' columns).")
    parser.add_argument("output_csv", help="Path for the output merged CSV file.")
    
    args = parser.parse_args()

    merge_csv_files(args.primary_csv, args.secondary_csv, args.output_csv)

if __name__ == "__main__":
    main()

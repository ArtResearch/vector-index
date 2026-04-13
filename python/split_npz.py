import numpy as np
import argparse
import os
import sys
import csv

def split_npz_file(input_file, output_folder):
    """
    Loads a .npz file, splits its arrays, and saves them into new files.
    One .npy file for embeddings and one .csv file for all other arrays.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Load the original .npz file
        print(f"Loading data from '{input_file}'...")
        with np.load(input_file, allow_pickle=True) as data:
            # Check for 'embeddings' array
            if 'embeddings' not in data:
                print(f"Error: 'embeddings' array not found in '{input_file}'.", file=sys.stderr)
                return
            
            embeddings = data['embeddings']
            
            # Identify all other arrays as metadata
            metadata_keys = [key for key in data.files if key != 'embeddings']
            
            if not metadata_keys:
                print(f"Warning: No metadata arrays found in '{input_file}'. Only saving embeddings.", file=sys.stderr)
            
            # Get the base name of the input file to construct output filenames
            base_name = os.path.basename(input_file)
            name, ext = os.path.splitext(base_name)

            # Define output file paths
            embeddings_output_path = os.path.join(output_folder, f"{name}-embeddings.npy")
            metadata_output_path = os.path.join(output_folder, f"{name}-metadata.csv")

            # Save the embeddings array to a .npy file
            print(f"Saving embeddings to '{embeddings_output_path}'...")
            np.save(embeddings_output_path, embeddings)

            # Save all metadata arrays to a single .csv file
            if metadata_keys:
                print(f"Saving metadata ({', '.join(metadata_keys)}) to '{metadata_output_path}'...")
                
                # Prepare data for writing: list of columns
                columns = [data[key] for key in metadata_keys]
                
                # Transpose columns to rows
                rows = zip(*columns)
                
                with open(metadata_output_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header
                    writer.writerow(metadata_keys)
                    # Write data rows
                    writer.writerows(rows)

            print("Splitting completed successfully.")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Split a .npz file with embeddings and URIs/texts into a .npy file for embeddings and a .csv file for URIs/texts.")
    parser.add_argument("input_file", help="Path to the input .npz file.")
    parser.add_argument("output_folder", help="Path to the folder where the output files will be saved.")
    args = parser.parse_args()

    split_npz_file(args.input_file, args.output_folder)

if __name__ == "__main__":
    main()

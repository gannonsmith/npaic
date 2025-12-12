import pandas as pd
import json
import argparse
import sys

def convert_jsonl_to_csv(input_path, output_path):
    """
    Reads a JSONL file line by line, converts each valid JSON object
    into a dictionary, and then creates a pandas DataFrame, which is
    saved to a CSV file.
    
    This method is memory-efficient for large files and robust to
    malformed JSON lines.
    """
    print(f"Starting conversion from {input_path} to {output_path}...")
    
    data = []
    malformed_lines = 0
    
    try:
        # Open the JSONL file for reading
        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                # Strip whitespace and check if line is empty
                line = line.strip()
                if not line:
                    print(f"Warning: Skipping empty line at position {i+1}.")
                    continue
                    
                try:
                    # Try to parse the line as JSON
                    record = json.loads(line)
                    
                    # We expect each line to be a JSON object (dictionary)
                    if isinstance(record, dict):
                        data.append(record)
                    else:
                        print(f"Warning: Line {i+1} is valid JSON but not an object (dict). Skipping.")
                        malformed_lines += 1
                        
                except json.JSONDecodeError:
                    # Handle lines that are not valid JSON
                    print(f"Warning: Skipping malformed JSON on line {i+1}: {line[:100]}...")
                    malformed_lines += 1
    
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)

    # Check if we actually found any valid data
    if not data:
        print("Error: No valid JSON objects were found in the file.")
        sys.exit(1)

    try:
        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(data)
        
        # Save the DataFrame to a CSV file
        # index=False prevents pandas from writing the DataFrame row index
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        print("\n--- Conversion Summary ---")
        print(f"Successfully converted {len(data)} records.")
        if malformed_lines > 0:
            print(f"Skipped {malformed_lines} malformed or invalid lines.")
        print(f"Output saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred while creating or saving the CSV: {e}")
        sys.exit(1)

if __name__ == "__main__":
    """
    Main entry point for the script.
    Parses command-line arguments for input and output file paths.
    """
    
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Convert a JSONL (.jsonl) file to a CSV (.csv) file.",
        epilog="Example: python jsonl_to_csv.py my_data.jsonl output.csv"
    )
    
    # Add arguments
    parser.add_argument(
        "input",
        help="The path to the input .jsonl file."
    )
    parser.add_argument(
        "output",
        help="The path for the output .csv file."
    )
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Run the conversion function
    convert_jsonl_to_csv(args.input, args.output)
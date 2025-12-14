import json

def trim_predicted_response(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line)
            response = data.get("predicted_response", "")
            # Find the position of the first "###"
            index = response.find("###")
            if index != -1:
                # Keep only the part before "###"
                data["predicted_response"] = response[:index].rstrip()
            # Write back the modified data
            outfile.write(json.dumps(data) + "\n")

if __name__ == "__main__":
    input_file = "results/predictions/finetuned_model/predictions.jsonl"   # Replace with your input file path
    output_file = "results/predictions/finetuned_model/cleaned_predictions.jsonl" # Replace with your desired output file path
    trim_predicted_response(input_file, output_file)

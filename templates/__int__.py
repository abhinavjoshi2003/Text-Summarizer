import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained('t5-small')  # Or t5-base/t5-large for larger models
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Define input and output folders
input_folder = 'input_texts/'    # Folder containing multiple text files
output_folder = 'summarized_texts/'  # Folder to save summarized text files

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to summarize text using T5
def summarize_text(text):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Iterate over all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):  # Process only .txt files
        input_path = os.path.join(input_folder, filename)
        
        # Read the content of the text file
        with open(input_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Summarize the text content
        summary = summarize_text(text)

        # Save the summary in a new file in the output folder
        output_filename = f"summary_{filename}"  # Prefix the original filename with "summary_"
        output_path = os.path.join(output_folder, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(summary)
        
        print(f"Summarized {filename} and saved to {output_path}")

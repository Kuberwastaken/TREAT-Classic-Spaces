import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import gradio as gr

# Fetch the Hugging Face token from the environment variable (secrets)
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN environment variable is not set!")

def analyze_script(script):
    # Starting the script analysis
    print("\n=== Starting Analysis ===")
    print(f"Time: {datetime.now()}")  # Outputting the current timestamp
    print("Loading model and tokenizer...")

    try:
        # Load the tokenizer and model, selecting the appropriate device (CPU or CUDA)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_fast=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"  # Use CUDA if available, else use CPU
        print(f"Using device: {device}")

        # Load model with token authentication
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            use_auth_token=hf_token,  # Pass the token to authenticate
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # Use 16-bit precision for CUDA, 32-bit for CPU
            device_map="auto"  # Automatically map model to available device
        )
        print("Model loaded successfully")

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    # Define trigger categories with their descriptions
    trigger_categories = {
        "Violence": {"mapped_name": "Violence", "description": "Any act involving physical force or aggression intended to cause harm, injury, or death to a person, animal, or object."},
        "Death": {"mapped_name": "Death References", "description": "Any mention, implication, or depiction of the loss of life."},
        "Substance Use": {"mapped_name": "Substance Use", "description": "Any explicit or implied reference to the consumption, misuse, or abuse of drugs, alcohol, or other intoxicating substances."},
        "Gore": {"mapped_name": "Gore", "description": "Extreme, graphic depictions of bodily harm, injury, mutilation, or bloodshed."},
        "Vomit": {"mapped_name": "Vomit", "description": "References to vomiting, nausea, or aftermath of vomiting."},
        "Sexual Content": {"mapped_name": "Sexual Content", "description": "Any depiction or mention of sexual activity or behavior."},
        "Sexual Abuse": {"mapped_name": "Sexual Abuse", "description": "Any form of non-consensual sexual act or behavior, including coercion or manipulation."},
        "Self-Harm": {"mapped_name": "Self-Harm", "description": "Behaviors where an individual intentionally causes harm to themselves, including suicidal ideation."},
        "Gun Use": {"mapped_name": "Gun Use", "description": "References to firearms being handled or used in a threatening manner."},
        "Animal Cruelty": {"mapped_name": "Animal Cruelty", "description": "Any harm or abuse toward animals, whether intentional or accidental."},
        "Mental Health Issues": {"mapped_name": "Mental Health Issues", "description": "References to mental health struggles, including depression, anxiety, PTSD, etc."}
    }

    print("\nProcessing text...")  # Output indicating the text is being processed
    chunk_size = 256  # Set the chunk size for text processing
    overlap = 15  # Overlap between chunks for context preservation
    script_chunks = [script[i:i + chunk_size] for i in range(0, len(script), chunk_size - overlap)]

    identified_triggers = {}

    for chunk_idx, chunk in enumerate(script_chunks, 1):
        print(f"\n--- Processing Chunk {chunk_idx}/{len(script_chunks)} ---")
        for category, info in trigger_categories.items():
            mapped_name = info["mapped_name"]
            description = info["description"]

            print(f"\nAnalyzing for {mapped_name}...")
            prompt = f"""
            Check this text for any indication of {mapped_name} ({description}).
            Be sensitive to subtle references or implications, make sure the text is not metaphorical.
            Respond concisely with: YES, NO, or MAYBE.
            Text: {chunk}
            Answer:
            """

            print(f"Sending prompt to model...")  # Indicate that prompt is being sent to the model
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)  # Tokenize the prompt
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Send inputs to the chosen device

            with torch.no_grad():  # Disable gradient calculation for inference
                print("Generating response...")  # Indicate that the model is generating a response
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=3,  # Limit response length
                    do_sample=True,  # Enable sampling for more diverse output
                    temperature=0.5,  # Control randomness of the output
                    top_p=0.9,  # Use nucleus sampling
                    pad_token_id=tokenizer.eos_token_id  # Pad token ID
                )
            
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().upper()  # Decode and format the response
            first_word = response_text.split("\n")[-1].split()[0] if response_text else "NO"  # Get the first word of the response
            print(f"Model response for {mapped_name}: {first_word}")

            # Update identified triggers based on model response
            if first_word == "YES":
                print(f"Detected {mapped_name} in this chunk!")  # Trigger detected
                identified_triggers[mapped_name] = identified_triggers.get(mapped_name, 0) + 1
            elif first_word == "MAYBE":
                print(f"Possible {mapped_name} detected, marking for further review.")  # Possible trigger detected
                identified_triggers[mapped_name] = identified_triggers.get(mapped_name, 0) + 0.5
            else:
                print(f"No {mapped_name} detected in this chunk.")  # No trigger detected

    print("\n=== Analysis Complete ===")  # Indicate that analysis is complete
    final_triggers = []  # List to store final triggers

    # Filter and output the final trigger results
    for mapped_name, count in identified_triggers.items():
        if count > 0.5:
            final_triggers.append(mapped_name)
        print(f"- {mapped_name}: found in {count} chunks")

    if not final_triggers:
        final_triggers = ["None"]

    return final_triggers

# Define the Gradio interface
def analyze_content(script):
    # Perform the analysis on the input script using the analyze_script function
    triggers = analyze_script(script)

    # Define the result based on the triggers found
    if isinstance(triggers, list) and triggers != ["None"]:
        result = {
            "detected_triggers": triggers,
            "confidence": "High - Content detected",
            "model": "Llama-3.2-1B",
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    else:
        result = {
            "detected_triggers": ["None"],
            "confidence": "High - No concerning content detected",
            "model": "Llama-3.2-1B",
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    print("\nFinal Result Dictionary:", result)
    return result

# Create and launch the Gradio interface
iface = gr.Interface(
    fn=analyze_content,
    inputs=gr.Textbox(lines=8, label="Input Text"),
    outputs=gr.JSON(),
    title="Content Analysis",
    description="Analyze text content for sensitive topics"
)

if __name__ == "__main__":
    iface.launch()

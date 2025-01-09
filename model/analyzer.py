# analyzer.py
# model > analyzer.py

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
            token=hf_token,  # Pass the token to authenticate
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # Use 16-bit precision for CUDA, 32-bit for CPU
            device_map="auto"  # Automatically map model to available device
        )
        print("Model loaded successfully")

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    # Define trigger categories with their descriptions
    trigger_categories = {
        "Violence": {
            "mapped_name": "Violence",
            "description": (
                "Any act involving physical force or aggression intended to cause harm, injury, or death to a person, animal, or object. "
                "Includes direct physical confrontations (e.g., fights, beatings, or assaults), implied violence (e.g., very graphical threats or descriptions of injuries), "
                "or large-scale events like wars, riots, or violent protests."
            )
        },
        "Death": {
            "mapped_name": "Death References",
            "description": (
                "Any mention, implication, or depiction of the loss of life, including direct deaths of characters, including mentions of deceased individuals, "
                "or abstract references to mortality (e.g., 'facing the end' or 'gone forever'). This also covers depictions of funerals, mourning, "
                "grieving, or any dialogue that centers around death, do not take metaphors into context that don't actually lead to death."
            )
        },
        "Substance Use": {
            "mapped_name": "Substance Use",
            "description": (
                "Any explicit or implied reference to the consumption, misuse, or abuse of drugs, alcohol, or other intoxicating substances. "
                "Includes scenes of drinking, smoking, or drug use, whether recreational or addictive. May also cover references to withdrawal symptoms, "
                "rehabilitation, or substance-related paraphernalia (e.g., needles, bottles, pipes)."
            )
        },
        "Gore": {
            "mapped_name": "Gore",
            "description": (
                "Extremely detailed and graphic depictions of highly severe physical injuries, mutilation, or extreme bodily harm, often accompanied by descriptions of heavy blood, exposed organs, "
                "or dismemberment. This includes war scenes with severe casualties, horror scenarios involving grotesque creatures, or medical procedures depicted with excessive detail."
            )
        },
        "Vomit": {
            "mapped_name": "Vomit",
            "description": (
                "Any reference to the act of vomiting, whether directly described, implied, or depicted in detail. This includes sounds or visual descriptions of the act, "
                "mentions of nausea leading to vomiting, or its aftermath (e.g., the presence of vomit, cleaning it up, or characters reacting to it)."
            )
        },
        "Sexual Content": {
            "mapped_name": "Sexual Content",
            "description": (
                "Any depiction or mention of sexual activity, intimacy, or sexual behavior, ranging from implied scenes to explicit descriptions. "
                "This includes romantic encounters, physical descriptions of characters in a sexual context, sexual dialogue, or references to sexual themes (e.g., harassment, innuendos)."
            )
        },
        "Sexual Abuse": {
            "mapped_name": "Sexual Abuse",
            "description": (
                "Any form of non-consensual sexual act, behavior, or interaction, involving coercion, manipulation, or physical force. "
                "This includes incidents of sexual assault, molestation, exploitation, harassment, and any acts where an individual is subjected to sexual acts against their will or without their consent. "
                "It also covers discussions or depictions of the aftermath of such abuse, such as trauma, emotional distress, legal proceedings, or therapy. "
                "References to inappropriate sexual advances, groping, or any other form of sexual misconduct are also included, as well as the psychological and emotional impact on survivors. "
                "Scenes where individuals are placed in sexually compromising situations, even if not directly acted upon, may also fall under this category."
            )
        },
        "Self-Harm": {
            "mapped_name": "Self-Harm",
            "description": (
                "Any mention or depiction of behaviors where an individual intentionally causes harm to themselves. This includes cutting, burning, or other forms of physical injury, "
                "as well as suicidal ideation, suicide attempts, or discussions of self-destructive thoughts and actions. References to scars, bruises, or other lasting signs of self-harm are also included."
            )
        },
        "Gun Use": {
            "mapped_name": "Gun Use",
            "description": (
                "Any explicit or implied mention of firearms being handled, fired, or used in a threatening manner. This includes scenes of gun violence, references to shootings, "
                "gun-related accidents, or the presence of firearms in a tense or dangerous context (e.g., holstered weapons during an argument)."
            )
        },
        "Animal Cruelty": {
            "mapped_name": "Animal Cruelty",
            "description": (
                "Any act of harm, abuse, or neglect toward animals, whether intentional or accidental. This includes physical abuse (e.g., hitting, injuring, or killing animals), "
                "mental or emotional mistreatment (e.g., starvation, isolation), and scenes where animals are subjected to pain or suffering for human entertainment or experimentation."
            )
        },
        "Mental Health Issues": {
            "mapped_name": "Mental Health Issues",
            "description": (
                "Any reference to mental health struggles, disorders, or psychological distress. This includes mentions of depression, anxiety, PTSD, bipolar disorder, schizophrenia, "
                "or other conditions. Scenes depicting therapy sessions, psychiatric treatment, or coping mechanisms (e.g., medication, journaling) are also included. May cover subtle hints "
                "like a character expressing feelings of worthlessness, hopelessness, or detachment from reality."
            )
        }
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

            print("Sending prompt to model...")  # Indicate that prompt is being sent to the model
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)  # Tokenize the prompt
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Send inputs to the chosen device

            with torch.no_grad():  # Disable gradient calculation for inference
                print("Generating response...")  # Indicate that the model is generating a response
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=3,  # Limit response length
                    do_sample=True,  # Enable sampling for more diverse output
                    temperature=0.7,  # Control randomness of the output
                    top_p=0.8,  # Use nucleus sampling
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
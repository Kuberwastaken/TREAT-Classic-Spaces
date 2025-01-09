import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import gradio as gr

# Fetch the Hugging Face token from the environment variable (secrets)
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN environment variable is not set!")

# Define trigger categories
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
    # Add other trigger categories here
}

def analyze_script(script):
    print("\n=== Starting Analysis ===")
    print(f"Time: {datetime.now()}")

    try:
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_fast=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            token=hf_token,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"An error occurred while loading model: {e}")
        return []

    print("\nProcessing text...")
    chunk_size = 256
    overlap = 15
    script_chunks = [script[i:i + chunk_size] for i in range(0, len(script), chunk_size - overlap)]

    identified_triggers = {}

    for chunk_idx, chunk in enumerate(script_chunks, 1):
        print(f"\n--- Processing Chunk {chunk_idx}/{len(script_chunks)} ---")
        for category, info in trigger_categories.items():
            mapped_name = info["mapped_name"]
            description = info["description"]

            prompt = f"""
            Check this text for any indication of {mapped_name} ({description}).
            Be sensitive to subtle references or implications, make sure the text is not metaphorical.
            Respond concisely with: YES, NO, or MAYBE.
            Text: {chunk}
            Answer:
            """

            print(f"Analyzing for {mapped_name}...")
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=3,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    pad_token_id=tokenizer.eos_token_id
                )

            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().upper()
            first_word = response_text.split("\n")[-1].split()[0] if response_text else "NO"
            print(f"Model response for {mapped_name}: {first_word}")

            if first_word == "YES":
                identified_triggers[mapped_name] = identified_triggers.get(mapped_name, 0) + 1
            elif first_word == "MAYBE":
                identified_triggers[mapped_name] = identified_triggers.get(mapped_name, 0) + 0.5

    final_triggers = [k for k, v in identified_triggers.items() if v > 0.5]
    if not final_triggers:
        final_triggers = ["None"]

    return final_triggers

def analyze_content(script):
    triggers = analyze_script(script)
    result = {
        "detected_triggers": triggers,
        "confidence": "High - Content detected" if triggers != ["None"] else "High - No concerning content detected",
        "model": "Llama-3.2-1B",
        "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    print("\nFinal Result Dictionary:", result)
    return result

# Gradio interface
iface = gr.Interface(
    fn=analyze_content,
    inputs=gr.Textbox(lines=8, label="Input Text"),
    outputs=gr.JSON(),
    title="Content Analysis",
    description="Analyze text content for triggers like violence, death, and more."
)

if __name__ == "__main__":
    iface.launch()

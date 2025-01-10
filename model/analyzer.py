import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import gradio as gr
from typing import Dict, List, Union, Optional
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentAnalyzer:
    def __init__(self):
        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable is not set!")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        logger.info(f"Initialized analyzer with device: {self.device}")

    async def load_model(self, progress=None) -> None:
        """Load the model and tokenizer with progress updates and detailed logging."""
        try:
            print("\n=== Starting Model Loading ===")
            print(f"Time: {datetime.now()}")
            
            if progress:
                progress(0.1, "Loading tokenizer...")
            
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B",
                use_fast=True
            )

            if progress:
                progress(0.3, "Loading model...")
            
            print(f"Loading model on {self.device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-1B",
                token=self.hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )

            if progress:
                progress(0.5, "Model loaded successfully")
            
            print("Model and tokenizer loaded successfully")
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            print(f"\nERROR DURING MODEL LOADING: {str(e)}")
            print("Stack trace:")
            traceback.print_exc()
            raise

    def _chunk_text(self, text: str, chunk_size: int = 256, overlap: int = 15) -> List[str]:
        """Split text into overlapping chunks for processing."""
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        print(f"Split text into {len(chunks)} chunks with {overlap} token overlap")
        return chunks

    async def analyze_chunk(
        self,
        chunk: str,
        trigger_categories: Dict,
        progress: Optional[gr.Progress] = None,
        current_progress: float = 0,
        progress_step: float = 0
    ) -> Dict[str, float]:
        """Analyze a single chunk of text for triggers with detailed logging."""
        chunk_triggers = {}
        print(f"\n--- Processing Chunk ---")
        print(f"Chunk text (preview): {chunk[:50]}...")
        
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

            try:
                print("Sending prompt to model...")
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    print("Generating response...")
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=5,
                        do_sample=True,
                        temperature=0.3,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().upper()
                first_word = response_text.split("\n")[-1].split()[0] if response_text else "NO"
                print(f"Model response for {mapped_name}: {first_word}")

                if first_word == "YES":
                    print(f"Detected {mapped_name} in this chunk!")
                    chunk_triggers[mapped_name] = chunk_triggers.get(mapped_name, 0) + 1
                elif first_word == "MAYBE":
                    print(f"Possible {mapped_name} detected, marking for further review.")
                    chunk_triggers[mapped_name] = chunk_triggers.get(mapped_name, 0) + 0.5
                else:
                    print(f"No {mapped_name} detected in this chunk.")

                if progress:
                    current_progress += progress_step
                    progress(min(current_progress, 0.9), f"Analyzing {mapped_name}...")

            except Exception as e:
                logger.error(f"Error analyzing chunk for {mapped_name}: {str(e)}")
                print(f"Error during analysis of {mapped_name}: {str(e)}")
                traceback.print_exc()

        return chunk_triggers

    async def analyze_script(self, script: str, progress: Optional[gr.Progress] = None) -> List[str]:
        """Analyze the entire script for triggers with progress updates and detailed logging."""
        print("\n=== Starting Script Analysis ===")
        print(f"Time: {datetime.now()}")

        if not self.model or not self.tokenizer:
            await self.load_model(progress)

        # Initialize trigger categories (kept from your working script)
        trigger_categories = {
            "Violence": {
                "mapped_name": "Violence",
                "description": (
                    "Any act involving physical force or aggression intended to cause harm, injury, or death to a person, animal, or object. "
                    "Includes direct physical confrontations (e.g., fights, beatings, or assaults), implied violence (e.g., very graphical threats or descriptions of injuries), "
                    "or large-scale events like wars, riots, or violent protests."
                )
            },
            # ... [other categories remain the same]
        }

        chunks = self._chunk_text(script)
        identified_triggers = {}
        progress_step = 0.4 / (len(chunks) * len(trigger_categories))
        current_progress = 0.5  # Starting after model loading

        for chunk_idx, chunk in enumerate(chunks, 1):
            chunk_triggers = await self.analyze_chunk(
                chunk,
                trigger_categories,
                progress,
                current_progress,
                progress_step
            )
            
            for trigger, count in chunk_triggers.items():
                identified_triggers[trigger] = identified_triggers.get(trigger, 0) + count

        if progress:
            progress(0.95, "Finalizing results...")

        print("\n=== Analysis Complete ===")
        print("Final Results:")
        final_triggers = []

        for mapped_name, count in identified_triggers.items():
            if count > 0.5:
                final_triggers.append(mapped_name)
            print(f"- {mapped_name}: found in {count} chunks")

        if not final_triggers:
            print("No triggers detected")
            final_triggers = ["None"]

        return final_triggers

async def analyze_content(
    script: str,
    progress: Optional[gr.Progress] = None
) -> Dict[str, Union[List[str], str]]:
    """Main analysis function for the Gradio interface with detailed logging."""
    print("\n=== Starting Content Analysis ===")
    print(f"Time: {datetime.now()}")
    
    analyzer = ContentAnalyzer()
    
    try:
        triggers = await analyzer.analyze_script(script, progress)
        
        if progress:
            progress(1.0, "Analysis complete!")

        result = {
            "detected_triggers": triggers,
            "confidence": "High - Content detected" if triggers != ["None"] else "High - No concerning content detected",
            "model": "Llama-3.2-1B",
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        print("\nFinal Result Dictionary:", result)
        return result

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        print(f"\nERROR OCCURRED: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        return {
            "detected_triggers": ["Error occurred during analysis"],
            "confidence": "Error",
            "model": "Llama-3.2-1B",
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(e)
        }

if __name__ == "__main__":
    # Gradio interface
    iface = gr.Interface(
        fn=analyze_content,
        inputs=gr.Textbox(lines=8, label="Input Text"),
        outputs=gr.JSON(),
        title="Content Analysis",
        description="Analyze text content for sensitive topics"
    )
    iface.launch()
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
                "meta-llama/Llama-3.2-3B",
                use_fast=True
            )

            if progress:
                progress(0.3, "Loading model...")
            
            print(f"Loading model on {self.device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-3B",
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

    def _chunk_text(self, text: str, chunk_size: int = 256, overlap: int = 20) -> List[str]:
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
            Check this text for any clear indication of {mapped_name} ({description}).
            only say yes if you are confident, make sure the text is not metaphorical.
            Respond concisely and only with: YES, NO, or MAYBE.
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
                        max_new_tokens=2,
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
                            "Any act of physical force meant to cause harm, injury, or death, including fights, threats, and large-scale violence like wars or riots."
                        )
                    },

            "Death": {
                        "mapped_name": "Death References",
                        "description": (
                            "Mentions or depictions of death, such as characters dying, references to deceased people, funerals, or mourning."
                        )
                    },

            "Substance Use": {
                        "mapped_name": "Substance Use",
                        "description": (
                            "Any reference to using or abusing drugs, alcohol, or other substances, including scenes of drinking, smoking, or drug use."
                        )
                    },

            "Gore": {
                        "mapped_name": "Gore",
                        "description": (
                            "Graphic depictions of severe injuries or mutilation, often with detailed blood, exposed organs, or dismemberment."
                        )
                    },

            "Vomit": {
                        "mapped_name": "Vomit",
                        "description": (
                            "Any explicit reference to vomiting or related actions. This includes only very specific mentions of nausea or the act of vomiting, with more focus on the direct description, only flag this if you absolutely believe it's present."
                        )
                    },

            "Sexual Content": {
                        "mapped_name": "Sexual Content",
                        "description": (
                            "Depictions or mentions of sexual activity, intimacy, or behavior, including sexual themes like harassment or innuendo."
                        )
                    },

            "Sexual Abuse": {
                        "mapped_name": "Sexual Abuse",
                        "description": (
                            "Explicit non-consensual sexual acts, including assault, molestation, or harassment, and the emotional or legal consequences of such abuse. A stronger focus on detailed depictions or direct references to coercion or violence."
                        )
                    },

            "Self-Harm": {
                        "mapped_name": "Self-Harm",
                        "description": (
                            "Depictions or mentions of intentional self-injury, including acts like cutting, burning, or other self-destructive behavior. Emphasis on more graphic or repeated actions, not implied or casual references."
                        )
                    },

            "Gun Use": {
                        "mapped_name": "Gun Use",
                        "description": (
                            "Explicit mentions of firearms in use, including threatening actions or accidents involving guns. Only triggers when the gun use is shown in a clear, violent context."
                        )
                    },

            "Animal Cruelty": {
                        "mapped_name": "Animal Cruelty",
                        "description": (
                            "Direct or explicit harm, abuse, or neglect of animals, including physical abuse or suffering, and actions performed for human entertainment or experimentation. Triggers only in clear, violent depictions."
                        )
                    },

            "Mental Health Issues": {
                        "mapped_name": "Mental Health Issues",
                        "description": (
                            "References to psychological struggles, such as depression, anxiety, or PTSD, including therapy or coping mechanisms."
                        )
                    }
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
            "model": "Llama-3.2-3B",
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
            "model": "Llama-3.2-3B",
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
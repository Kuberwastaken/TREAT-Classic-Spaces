import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import gradio as gr
from typing import Dict, List, Union, Optional
import logging

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
        self.trigger_categories = self._init_trigger_categories()

    def _init_trigger_categories(self) -> Dict:
        """Initialize trigger categories with their descriptions."""
        return {
            "Violence": {
                "mapped_name": "Violence",
                "description": (
                    "Any act involving physical force or aggression intended to cause harm, injury, or death to a person, animal, or object. "
                    "Includes direct physical confrontations, implied violence, or large-scale events like wars, riots, or violent protests."
                )
            },
            "Death": {
                "mapped_name": "Death References",
                "description": (
                    "Any mention, implication, or depiction of the loss of life, including direct deaths of characters, mentions of deceased individuals, "
                    "or abstract references to mortality. This covers depictions of funerals, mourning, or death-centered dialogue."
                )
            },
            "Substance Use": {
                "mapped_name": "Substance Use",
                "description": (
                    "Any explicit or implied reference to the consumption, misuse, or abuse of drugs, alcohol, or other intoxicating substances. "
                    "Includes scenes of drinking, smoking, drug use, withdrawal symptoms, or rehabilitation."
                )
            },
            "Gore": {
                "mapped_name": "Gore",
                "description": (
                    "Extremely detailed and graphic depictions of severe physical injuries, mutilation, or extreme bodily harm, including heavy blood, "
                    "exposed organs, or dismemberment."
                )
            },
            "Vomit": {
                "mapped_name": "Vomit",
                "description": "Any reference to the act of vomiting, whether directly described, implied, or depicted in detail."
            },
            "Sexual Content": {
                "mapped_name": "Sexual Content",
                "description": (
                    "Any depiction or mention of sexual activity, intimacy, or sexual behavior, from implied scenes to explicit descriptions."
                )
            },
            "Sexual Abuse": {
                "mapped_name": "Sexual Abuse",
                "description": (
                    "Any form of non-consensual sexual act, behavior, or interaction, involving coercion, manipulation, or physical force."
                )
            },
            "Self-Harm": {
                "mapped_name": "Self-Harm",
                "description": (
                    "Any mention or depiction of behaviors where an individual intentionally causes harm to themselves, including suicidal thoughts."
                )
            },
            "Gun Use": {
                "mapped_name": "Gun Use",
                "description": (
                    "Any explicit or implied mention of firearms being handled, fired, or used in a threatening manner."
                )
            },
            "Animal Cruelty": {
                "mapped_name": "Animal Cruelty",
                "description": (
                    "Any act of harm, abuse, or neglect toward animals, whether intentional or accidental."
                )
            },
            "Mental Health Issues": {
                "mapped_name": "Mental Health Issues",
                "description": (
                    "Any reference to mental health struggles, disorders, or psychological distress, including therapy and treatment."
                )
            }
        }

    async def load_model(self, progress=None) -> None:
        """Load the model and tokenizer with progress updates."""
        try:
            if progress:
                progress(0.1, "Loading tokenizer...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-1B",
                use_fast=True
            )

            if progress:
                progress(0.3, "Loading model...")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-1B",
                token=self.hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )

            if progress:
                progress(0.5, "Model loaded successfully")
            
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _chunk_text(self, text: str, chunk_size: int = 256, overlap: int = 15) -> List[str]:
        """Split text into overlapping chunks for processing."""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

    async def analyze_chunk(
        self,
        chunk: str,
        progress: Optional[gr.Progress] = None,
        current_progress: float = 0,
        progress_step: float = 0
    ) -> Dict[str, float]:
        """Analyze a single chunk of text for triggers."""
        chunk_triggers = {}
        
        for category, info in self.trigger_categories.items():
            mapped_name = info["mapped_name"]
            description = info["description"]

            prompt = f"""
            Check this text for any indication of {mapped_name} ({description}).
            Be sensitive to subtle references or implications, make sure the text is not metaphorical.
            Respond concisely with: YES, NO, or MAYBE.
            Text: {chunk}
            Answer:
            """

            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=True,
                        temperature=0.5,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().upper()
                first_word = response_text.split("\n")[-1].split()[0] if response_text else "NO"

                if first_word == "YES":
                    chunk_triggers[mapped_name] = chunk_triggers.get(mapped_name, 0) + 1
                elif first_word == "MAYBE":
                    chunk_triggers[mapped_name] = chunk_triggers.get(mapped_name, 0) + 0.5

                if progress:
                    current_progress += progress_step
                    progress(min(current_progress, 0.9), f"Analyzing {mapped_name}...")

            except Exception as e:
                logger.error(f"Error analyzing chunk for {mapped_name}: {str(e)}")

        return chunk_triggers

    async def analyze_script(self, script: str, progress: Optional[gr.Progress] = None) -> List[str]:
        """Analyze the entire script for triggers with progress updates."""
        if not self.model or not self.tokenizer:
            await self.load_model(progress)

        chunks = self._chunk_text(script)
        identified_triggers = {}
        progress_step = 0.4 / (len(chunks) * len(self.trigger_categories))
        current_progress = 0.5  # Starting after model loading

        for chunk_idx, chunk in enumerate(chunks, 1):
            chunk_triggers = await self.analyze_chunk(
                chunk,
                progress,
                current_progress,
                progress_step
            )
            
            for trigger, count in chunk_triggers.items():
                identified_triggers[trigger] = identified_triggers.get(trigger, 0) + count

        if progress:
            progress(0.95, "Finalizing results...")

        final_triggers = [
            trigger for trigger, count in identified_triggers.items()
            if count > 0.5
        ]

        return final_triggers if final_triggers else ["None"]

async def analyze_content(
    script: str,
    progress: Optional[gr.Progress] = None
) -> Dict[str, Union[List[str], str]]:
    """Main analysis function for the Gradio interface."""
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

        return result

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {
            "detected_triggers": ["Error occurred during analysis"],
            "confidence": "Error",
            "model": "Llama-3.2-1B",
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(e)
        }

if __name__ == "__main__":
    # This section is mainly for testing the analyzer directly
    iface = gr.Interface(
        fn=analyze_content,
        inputs=gr.Textbox(lines=8, label="Input Text"),
        outputs=gr.JSON(),
        title="Content Analysis",
        description="Analyze text content for sensitive topics"
    )
    iface.launch()
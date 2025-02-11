import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.batch_size = 4
        self.trigger_categories = {
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
            "Substance_Use": {
                "mapped_name": "Substance Use",
                "description": (
                    "Any explicit reference to the consumption, misuse, or abuse of drugs, alcohol, or other intoxicating substances. "
                    "This includes scenes of drug use, drinking, smoking, discussions about heavy substance abuse or substance-related paraphernalia."
                )
            },
            "Gore": {
                "mapped_name": "Gore",
                "description": (
                    "Extremely detailed and graphic depictions of highly severe physical injuries, mutilation, or extreme bodily harm, often accompanied by descriptions of heavy blood, exposed organs, "
                    "or dismemberment. This includes war scenes with severe casualties, horror scenarios involving grotesque creatures, or medical procedures depicted with excessive detail."
                )
            },
            "Sexual_Content": {
                "mapped_name": "Sexual Content",
                "description": (
                    "Any depiction of sexual activity, intimacy, or sexual behavior, ranging from implied scenes to explicit descriptions. "
                    "This includes physical descriptions of characters in a sexual context, sexual dialogue, or references to sexual themes."
                )
            },
            "Sexual_Abuse": {
               "mapped_name": "Sexual Abuse",
               "description": (
                  "Any form of non-consensual sexual act, behavior, or interaction, involving coercion, manipulation, or physical force. "
                  "This includes incidents of sexual assault, exploitation, harassment, and any acts where an individual is subjected to sexual acts against their will."
               )
            },
            "Self_Harm": {
                "mapped_name": "Self-Harm",
                "description": (
                    "Any mention or depiction of behaviors where an individual intentionally causes harm to themselves. This includes cutting, burning, or other forms of physical injury, "
                    "as well as suicidal ideation, suicide attempts, or discussions of self-destructive thoughts and actions."
                )
            },
            "Mental_Health": {
                "mapped_name": "Mental Health Issues",
                "description": (
                    "Any reference to extreme mental health struggles, disorders, or psychological distress. This includes depictions of depression, anxiety, PTSD, bipolar disorder, "
                    "or other conditions. Also includes toxic traits such as Gaslighting or other psycholgoical horrors"
                )
            }
        }
        logger.info(f"Initialized analyzer with device: {self.device}")

    async def load_model(self, progress=None) -> None:
        """Load the model and tokenizer with progress updates."""
        try:
            if progress:
                progress(0.1, "Loading tokenizer...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                "google/flan-t5-base",
                use_fast=True
            )
            
            if progress:
                progress(0.3, "Loading model...")
            
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/flan-t5-base",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )
            
            if self.device == "cuda":
                self.model.eval()
                torch.cuda.empty_cache()
                
            if progress:
                progress(0.5, "Model loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 30) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks

    def _validate_response(self, response: str) -> str:
        """Validate and clean model response."""
        valid_responses = {"YES", "NO", "MAYBE"}
        response = response.strip().upper()
        first_word = response.split()[0] if response else "NO"
        return first_word if first_word in valid_responses else "NO"

    async def analyze_chunks_batch(
        self,
        chunks: List[str],
        progress: Optional[gr.Progress] = None,
        current_progress: float = 0,
        progress_step: float = 0
    ) -> Dict[str, float]:
        """Analyze multiple chunks in batches."""
        all_triggers = {}
        
        for category, info in self.trigger_categories.items():
            mapped_name = info["mapped_name"]
            description = info["description"]
            
            for i in range(0, len(chunks), self.batch_size):
                batch_chunks = chunks[i:i + self.batch_size]
                prompts = []
                
                for chunk in batch_chunks:
                    prompt = f"""
                    Task: Analyze if this text contains {mapped_name}.
                    Context: {description}
                    Text: "{chunk}"
                    
                    Rules for analysis:
                    1. Only answer YES if there is clear, direct evidence
                    2. Answer NO if the content is ambiguous or metaphorical
                    3. Consider the severity and context
                    
                    Answer with ONLY ONE word: YES, NO, or MAYBE
                    """
                    prompts.append(prompt)

                try:
                    inputs = self.tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=20,
                            temperature=0.2,
                            top_p=0.85,
                            num_beams=3,
                            early_stopping=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            do_sample=True
                        )
                    
                    responses = [
                        self.tokenizer.decode(output, skip_special_tokens=True)
                        for output in outputs
                    ]
                    
                    for response in responses:
                        validated_response = self._validate_response(response)
                        if validated_response == "YES":
                            all_triggers[mapped_name] = all_triggers.get(mapped_name, 0) + 1
                        elif validated_response == "MAYBE":
                            all_triggers[mapped_name] = all_triggers.get(mapped_name, 0) + 0.5
                
                except Exception as e:
                    logger.error(f"Error processing batch for {mapped_name}: {str(e)}")
                    continue
                
                if progress:
                    current_progress += progress_step
                    progress(min(current_progress, 0.9), f"Analyzing {mapped_name}...")
                    
        return all_triggers

    async def analyze_script(self, script: str, progress: Optional[gr.Progress] = None) -> List[str]:
        """Analyze the entire script."""
        if not self.model or not self.tokenizer:
            await self.load_model(progress)
        
        chunks = self._chunk_text(script)
        identified_triggers = await self.analyze_chunks_batch(
            chunks,
            progress,
            current_progress=0.5,
            progress_step=0.4 / (len(chunks) * len(self.trigger_categories))
        )
        
        if progress:
            progress(0.95, "Finalizing results...")

        final_triggers = []
        chunk_threshold = max(1, len(chunks) * 0.1)
        
        for mapped_name, count in identified_triggers.items():
            if count >= chunk_threshold:
                final_triggers.append(mapped_name)

        return final_triggers if final_triggers else ["None"]

async def analyze_content(
    script: str,
    progress: Optional[gr.Progress] = None
) -> Dict[str, Union[List[str], str]]:
    """Main analysis function for the Gradio interface."""
    logger.info("Starting content analysis")
    
    analyzer = ContentAnalyzer()
    
    try:
        # Fix: Use the analyzer instance's method instead of undefined function
        triggers = await analyzer.analyze_script(script, progress)
        
        if progress:
            progress(1.0, "Analysis complete!")

        result = {
            "detected_triggers": triggers,
            "confidence": "High - Content detected" if triggers != ["None"] else "High - No concerning content detected",
            "model": "google/flan-t5-base",
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        logger.info(f"Analysis complete: {result}")
        return result

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return {
            "detected_triggers": ["Error occurred during analysis"],
            "confidence": "Error",
            "model": "google/flan-t5-base",
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(e)
        }

if __name__ == "__main__":
    iface = gr.Interface(
        fn=analyze_content,
        inputs=gr.Textbox(lines=8, label="Input Text"),
        outputs=gr.JSON(),
        title="Content Trigger Analysis",
        description="Analyze text content for sensitive topics and trigger warnings"
    )
iface.launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    share=True
    )
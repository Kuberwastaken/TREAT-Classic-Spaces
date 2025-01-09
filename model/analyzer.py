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
                progress(0.15, "Loading model...")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-1B",
                token=self.hf_token,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto"
            )

            if progress:
                progress(0.2, "Model loaded successfully")
            
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _chunk_text(self, text: str, chunk_size: int = 256, overlap: int = 32) -> List[str]:
        """Split text into overlapping chunks for processing."""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            if end < text_len:
                last_period = max(
                    text.rfind('. ', start, end),
                    text.rfind('\n', start, end)
                )
                if last_period > start:
                    end = last_period + 1
            
            chunks.append(text[start:end])
            start = end - overlap
            
        return chunks

    def _process_model_response(self, response_text: str) -> float:
        """Process model response and return a confidence score."""
        response = response_text.strip().upper()
        
        if "YES" in response:
            evidence_words = ["CLEAR", "DEFINITELY", "EXPLICIT", "STRONG"]
            return 1.0 if any(word in response for word in evidence_words) else 0.8
        elif "MAYBE" in response or "POSSIBLE" in response:
            return 0.5
        elif "NO" in response:
            return 0.0
        
        positive_indicators = ["PRESENT", "FOUND", "CONTAINS", "SHOWS", "INDICATES"]
        negative_indicators = ["ABSENT", "NONE", "NOTHING", "LACKS"]
        
        if any(indicator in response for indicator in positive_indicators):
            return 0.7
        elif any(indicator in response for indicator in negative_indicators):
            return 0.0
        
        return 0.0

    async def analyze_chunk(
        self,
        chunk: str,
        progress: Optional[gr.Progress] = None,
        current_progress: float = 0,
        progress_step: float = 0
    ) -> Dict[str, float]:
        """Analyze a single chunk of text for triggers."""
        chunk_triggers = {}
        progress_increment = progress_step / len(self.trigger_categories)
        
        for category, info in self.trigger_categories.items():
            mapped_name = info["mapped_name"]
            description = info["description"]

            prompt = f"""
            Analyze this text carefully for any indication of {mapped_name}.
            Context: {description}
            
            Guidelines:
            - Consider both explicit and implicit references
            - Ignore metaphorical or figurative language
            - Look for concrete evidence in the text
            
            Text to analyze: {chunk}
            
            Is there evidence of {mapped_name}? Respond with YES, NO, or MAYBE and briefly explain why.
            Answer:
            """

            try:
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=32,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.92,
                        top_k=50,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                confidence = self._process_model_response(response_text)
                
                if confidence > 0.5:
                    chunk_triggers[mapped_name] = chunk_triggers.get(mapped_name, 0) + confidence

                if progress:
                    current_progress += progress_increment
                    progress(min(current_progress, 0.9), f"Analyzing for {mapped_name}...")

            except Exception as e:
                logger.error(f"Error analyzing chunk for {mapped_name}: {str(e)}")

        return chunk_triggers

    async def analyze_script(self, script: str, progress: Optional[gr.Progress] = None) -> List[str]:
        """Analyze the entire script for triggers with progress updates."""
        if not self.model or not self.tokenizer:
            await self.load_model(progress)

        chunks = self._chunk_text(script)
        trigger_scores = {}
        
        # Calculate progress allocation
        analysis_progress = 0.7  # 70% of progress for analysis
        progress_per_chunk = analysis_progress / len(chunks)
        current_progress = 0.2  # Starting after model loading
        
        if progress:
            progress(current_progress, "Beginning content analysis...")

        for i, chunk in enumerate(chunks):
            chunk_triggers = await self.analyze_chunk(
                chunk,
                progress,
                current_progress,
                progress_per_chunk
            )
            
            for trigger, score in chunk_triggers.items():
                trigger_scores[trigger] = trigger_scores.get(trigger, 0) + score
            
            current_progress += progress_per_chunk
            if progress:
                chunk_number = i + 1
                progress(min(0.9, current_progress), 
                        f"Processing chunk {chunk_number}/{len(chunks)}...")

        if progress:
            progress(0.95, "Finalizing analysis...")

        # Normalize scores by number of chunks and apply threshold
        chunk_count = len(chunks)
        final_triggers = [
            trigger for trigger, score in trigger_scores.items()
            if score / chunk_count > 0.3
        ]

        return final_triggers if final_triggers else ["None"]

async def analyze_content(
    script: str,
    progress: Optional[gr.Progress] = None
) -> Dict[str, Union[List[str], str]]:
    """Main analysis function for the Gradio interface."""
    analyzer = ContentAnalyzer()
    
    try:
        if progress:
            progress(0.0, "Initializing analyzer...")
        
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
    # Gradio interface
    iface = gr.Interface(
        fn=analyze_content,
        inputs=gr.Textbox(lines=8, label="Input Text"),
        outputs=gr.JSON(),
        title="Content Analysis",
        description="Analyze text content for sensitive topics"
    )
    iface.launch()
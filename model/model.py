from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime
import gc

class ContentAnalyzer:
    def __init__(self):
        self.model_name = "meta-llama/Llama-3.2-1B"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load model with memory optimization"""
        try:
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
            
            print(f"Loading model on {self.device}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
            return True
        except Exception as e:
            print(f"Model loading error: {str(e)}")
            return False

    def cleanup(self):
        """Clean up GPU memory"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def analyze_chunk(self, chunk, category_info):
        """Analyze a single chunk of text for a specific trigger"""
        mapped_name = category_info["mapped_name"]
        description = category_info["description"]

        prompt = f"""Check this text for any indication of {mapped_name} ({description}).
        Be sensitive to subtle references or implications, make sure the text is not metaphorical.
        Respond concisely with: YES, NO, or MAYBE.
        Text: {chunk}
        Answer:"""

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

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().upper()
            first_word = response.split("\n")[-1].split()[0] if response else "NO"
            
            score = 1 if first_word == "YES" else 0.5 if first_word == "MAYBE" else 0
            return score, first_word
            
        except Exception as e:
            print(f"Chunk analysis error: {str(e)}")
            return 0, "NO"

    def analyze_text(self, text):
        """Main analysis function"""
        if not self.load_model():
            return {"error": "Model loading failed"}

        # Original trigger categories
        trigger_categories = {
            "Violence": {
                "mapped_name": "Violence",
                "description": "Any act involving physical force or aggression intended to cause harm, injury, or death."
            },
            "Death": {
                "mapped_name": "Death References",
                "description": "Any mention, implication, or depiction of the loss of life, including direct deaths or abstract references to mortality."
            },
            "Substance_Use": {
                "mapped_name": "Substance Use",
                "description": "References to consumption, misuse, or abuse of drugs, alcohol, or other intoxicating substances."
            },
            "Gore": {
                "mapped_name": "Gore",
                "description": "Graphic depictions of severe physical injuries, mutilation, or extreme bodily harm."
            },
            "Sexual_Content": {
                "mapped_name": "Sexual Content",
                "description": "Depictions or mentions of sexual activity, intimacy, or sexual behavior."
            },
            "Self_Harm": {
                "mapped_name": "Self-Harm",
                "description": "Behaviors where an individual intentionally causes harm to themselves."
            },
            "Mental_Health": {
                "mapped_name": "Mental Health Issues",
                "description": "References to mental health struggles, disorders, or psychological distress."
            }
        }

        try:
            # Optimize chunk processing
            chunk_size = 200  # Reduced chunk size for better memory management
            overlap = 10
            chunks = []
            
            # Create chunks with overlap
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                chunks.append(chunk)
            
            trigger_scores = {}
            trigger_occurrences = {}
            
            # Initialize tracking dictionaries
            for category, info in trigger_categories.items():
                trigger_scores[info["mapped_name"]] = 0
                trigger_occurrences[info["mapped_name"]] = []
            
            # Process all chunks for all categories
            for chunk_idx, chunk in enumerate(chunks):
                print(f"\nProcessing chunk {chunk_idx + 1}/{len(chunks)}")
                chunk_triggers = {}
                
                for category, info in trigger_categories.items():
                    score, response = self.analyze_chunk(chunk, info)
                    
                    if score > 0:
                        mapped_name = info["mapped_name"]
                        trigger_scores[mapped_name] += score
                        trigger_occurrences[mapped_name].append({
                            'chunk_idx': chunk_idx,
                            'response': response,
                            'score': score
                        })
                        print(f"Found {mapped_name} in chunk {chunk_idx + 1} (Response: {response})")
                
                # Cleanup after processing each chunk
                if self.device == "cuda":
                    self.cleanup()

            # Collect all triggers that meet the threshold
            detected_triggers = []
            for name, score in trigger_scores.items():
                if score >= 0.5:  # Threshold for considering a trigger as detected
                    occurrences = len(trigger_occurrences[name])
                    detected_triggers.append(name)
                    print(f"\nTrigger '{name}' detected in {occurrences} chunks with total score {score}")
            
            result = {
                "detected_triggers": detected_triggers if detected_triggers else ["None"],
                "confidence": "High - Content detected" if detected_triggers else "High - No concerning content detected",
                "model": self.model_name,
                "analysis_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "trigger_details": {
                    name: {
                        "total_score": trigger_scores[name],
                        "occurrences": trigger_occurrences[name]
                    } for name in detected_triggers if name != "None"
                }
            }

            return result

        except Exception as e:
            return {"error": str(e)}
        finally:
            self.cleanup()

def get_detailed_analysis(script):
    analyzer = ContentAnalyzer()
    return analyzer.analyze_text(script)
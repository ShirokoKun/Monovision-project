"""
Tier 1 Language Model: Flan-T5-Small (Fast Mode)
GPU resident, fast inference, ≤20 tokens
"""

import asyncio
import logging
import time
from typing import Dict, Any, Tuple, Optional
import torch

try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logging.error(f"Missing transformers dependency: {e}")
    TRANSFORMERS_AVAILABLE = False
    T5ForConditionalGeneration = None
    T5Tokenizer = None

from ..core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)

class FlanT5Processor:
    """
    Tier 1 Language Processor using Flan-T5-Small
    - GPU resident for fast inference
    - Optimized for ≤20 token responses
    - Used in Fast mode: BLIP + Flan-T5
    """
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        self.resource_manager = resource_manager or ResourceManager()
        
        # Model configuration - OPTIMIZED FOR CUDA
        self.model_name = "google/flan-t5-small"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = torch.cuda.is_available()  # Use FP16 only on CUDA
        self.max_tokens = 50  # Increased from 20 for better responses
        
        # Model instances
        self.tokenizer = None
        self.model = None
        
        # Performance tracking
        self.generation_count = 0
        self.total_generation_time = 0.0
        self.last_used = time.time()
        
        logger.info("⚡ Flan-T5 Processor (Tier 1) initialized")
    
    async def initialize(self):
        """Initialize Flan-T5 model (GPU resident)"""
        logger.info("🚀 Loading Flan-T5-Small (Tier 1: Fast Mode)")

        # Check if transformers is available
        if not TRANSFORMERS_AVAILABLE:
            logger.error("❌ Transformers library not available. Please install with: pip install transformers")
            self.model = None
            self.tokenizer = None
            raise ImportError("Transformers library is required for Flan-T5 but not installed")

        try:
            # Load tokenizer
            logger.info("📝 Loading Flan-T5 tokenizer...")
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.model_name,
                cache_dir="cache/hf"
            )
            logger.info("✅ Tokenizer loaded successfully")

            # Load model with optimization - CUDA OPTIMIZED
            logger.info("🤖 Loading Flan-T5 model...")
            model_kwargs = {
                "cache_dir": "cache/hf",
                "low_cpu_mem_usage": True
            }
            
            # Use torch_dtype parameter for model loading
            if self.use_fp16 and self.device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16
                logger.info("🎯 Using FP16 precision for CUDA acceleration")
            else:
                model_kwargs["torch_dtype"] = torch.float32

            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            ).to(self.device)

            logger.info(f"✅ Model loaded successfully on {self.device}")

            # Set to eval mode
            self.model.eval()
            logger.info("✅ Model set to evaluation mode")

            # Register memory allocation
            estimated_memory = 800  # MB for Flan-T5-Small
            self.resource_manager.register_model_allocation("Flan-T5", estimated_memory)
            logger.info(f"📊 Memory allocation registered: {estimated_memory}MB")

            logger.info(f"🎉 Flan-T5 initialization completed successfully!")

        except Exception as e:
            logger.error(f"❌ Error loading Flan-T5: {e}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            # Set to None to indicate failure
            self.model = None
            self.tokenizer = None
            raise

    async def generate_response(self, vision_results: Dict[str, Any], user_query: str, max_tokens: int = 20, context_prompt: Dict = None) -> Tuple[str, int]:
        """
        Generate fast response using Flan-T5 with enhanced prompt engineering
        
        Args:
            vision_results: Results from vision fusion layer
            user_query: User's question
            max_tokens: Maximum tokens to generate (≤20 for fast mode)
            context_prompt: Enhanced context-aware prompt (V3 feature)
            
        Returns:
            Tuple of (response_text, token_count)
        """
        start_time = time.time()
        self.last_used = time.time()
        
        try:
            # Check if transformers is available and model/tokenizer are loaded
            if not TRANSFORMERS_AVAILABLE:
                logger.error("❌ Transformers library not available")
                return "Transformers library is required but not installed. Please install with: pip install transformers", 0

            # Check if model and tokenizer are properly loaded
            if self.model is None or self.tokenizer is None:
                logger.error("❌ Flan-T5 model or tokenizer not loaded")
                return "Model not properly initialized. Please try again.", 0
            
            # Extract vision context
            caption = vision_results.get("caption", "")
            keywords = vision_results.get("keywords", [])
            clip_keywords = vision_results.get("clip_keywords", [])
            
            # Build enhanced prompt with role priming and context
            if context_prompt:
                prompt = self._build_context_aware_prompt(context_prompt, vision_results, user_query)
            else:
                prompt = self._build_enhanced_fast_prompt(caption, keywords + clip_keywords, user_query)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Adaptive generation config based on device (CUDA vs CPU)
            if self.device == "cuda":
                # CUDA-optimized configuration
                generation_config = {
                    "max_new_tokens": min(max_tokens, self.max_tokens),
                    "do_sample": False,  # Greedy decoding for stability
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "use_cache": True,
                    "num_beams": 1,      # Single beam for speed
                    "early_stopping": False,
                    "repetition_penalty": 1.2  # Slightly higher to prevent loops
                }
            else:
                # CPU-optimized configuration
                generation_config = {
                    "max_new_tokens": min(max_tokens, self.max_tokens),
                    "do_sample": False,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "use_cache": True,
                    "num_beams": 1,
                    "early_stopping": False,
                    "repetition_penalty": 1.2
                }
            
            # Generate response with device-specific optimization
            with torch.no_grad():
                if self.use_fp16 and self.device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = self.model.generate(
                            **inputs,
                            **generation_config
                        )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        **generation_config
                    )
            
            # Decode response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Enhanced response cleaning with natural phrasing variation
            response_text = self._clean_response(response_text, prompt)
            
            # POST-PROCESS WEAK RESPONSES WITH STRONGER FALLBACKS
            original_response = response_text
            response_text = self._enhance_weak_response(response_text, vision_results, user_query)
            
            # CRITICAL: Auto-retry mechanism for weak responses
            if len(response_text.split()) < 3 or any(pattern in response_text.lower() for pattern in ["unclear", "unknown", "unsure"]):
                logger.info(f"🔄 Auto-retry triggered for weak response: '{response_text[:30]}...'")
                
                # Retry with stronger, more detailed prompt
                stronger_prompt = f"""{prompt}

Please expand your response and make it detailed and conversational."""
                
                stronger_inputs = self.tokenizer(
                    stronger_prompt,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                # Use more creative settings for retry
                retry_config = {
                    "max_new_tokens": min(30, self.max_tokens),
                    "do_sample": True,       # Enable sampling for creativity
                    "temperature": 0.8,      # Higher temperature for variety
                    "top_p": 0.9,           # Nucleus sampling
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "use_cache": True,
                    "repetition_penalty": 1.3  # Higher to avoid repetition
                }
                
                try:
                    with torch.no_grad():
                        if self.use_fp16:
                            with torch.autocast(device_type="cuda"):
                                retry_outputs = self.model.generate(
                                    **stronger_inputs,
                                    **retry_config
                                )
                        else:
                            retry_outputs = self.model.generate(
                                **stronger_inputs,
                                **retry_config
                            )
                    
                    retry_response = self.tokenizer.decode(retry_outputs[0], skip_special_tokens=True)
                    retry_response = self._clean_response(retry_response, stronger_prompt)
                    
                    # Use retry response if it's better
                    if len(retry_response.split()) > len(response_text.split()) and len(retry_response.split()) >= 5:
                        logger.info(f"✅ Retry successful: '{retry_response[:30]}...'")
                        response_text = retry_response
                    else:
                        logger.info(f"⚠️ Retry didn't improve response, keeping original")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Retry failed: {e}")
            
            # If we still have weak patterns after retry, apply emergency fallback
            weak_patterns = ["unclear", "unknown", "unsure", "blurry", "dark", "nothing", "empty", "something"]
            if any(pattern in response_text.lower() for pattern in weak_patterns):
                logger.info(f"🚨 Emergency fallback triggered - weak pattern still present after retry")
                response_text = "I can see visual content in this image that I can analyze and discuss with you."
            
            # Log if weak response was detected and replaced
            if response_text != original_response:
                logger.info(f"🔧 Weak response detected and enhanced: '{original_response[:30]}...' -> '{response_text[:30]}...'")
            
            # Add natural phrasing variation to avoid robotic repetition
            response_text = self._vary_response_phrasing(response_text)
            
            # Count tokens in response
            response_tokens = self.tokenizer.encode(response_text)
            token_count = len(response_tokens)
            
            # Update performance metrics
            generation_time = time.time() - start_time
            self.generation_count += 1
            self.total_generation_time += generation_time
            
            logger.info(f"⚡ Flan-T5 generated {token_count} tokens in {generation_time:.2f}s")
            
            return response_text, token_count
            
        except Exception as e:
            logger.error(f"❌ Flan-T5 generation error: {e}")
            return f"Error generating response: {str(e)}", 0
    
    def _build_fast_prompt(self, caption: str, user_query: str) -> str:
        """
        Build optimized prompt for fast mode (Legacy method for backward compatibility)
        Focus on quick, direct responses about visual content
        """
        return self._build_enhanced_fast_prompt(caption, [], user_query)
    
    def _build_enhanced_fast_prompt(self, caption: str, keywords: list, user_query: str) -> str:
        """
        Build ultra-simple prompt for FLAN-T5-Small with strong role priming
        Enhanced with system prompting and user message wrapping
        """
        
        # Strong system prompt for role priming
        system_prompt = (
            "You are MonoVision AI, a conversational assistant. "
            "Answer in natural sentences. Do not be robotic or vague. "
            "Always provide clear, detailed, and human-like answers."
        )
        
        # Format user message with instructions wrapper
        def format_prompt_with_instructions(user_message: str) -> str:
            return f"You are a helpful AI assistant. Respond conversationally, be clear and complete.\n\nUser: {user_message}\nAssistant:"
        
        # Ultra-simple, concrete QA format with role priming
        if user_query.strip() and caption.strip():
            # User has specific question about an image - keep it simple but add role context
            formatted_prompt = f"""{system_prompt}

{format_prompt_with_instructions(f"{user_query} Image shows: {caption}")}"""
        elif user_query.strip():
            # Text-only mode: User asks a question but no image context - be conversational
            formatted_prompt = f"""{system_prompt}

{format_prompt_with_instructions(user_query)}"""
        elif caption.strip():
            # No specific question, describe image - direct format with role
            formatted_prompt = f"""{system_prompt}

{format_prompt_with_instructions(f"Describe this image: {caption}")}"""
        else:
            # Fallback for no context - simplest possible with role
            formatted_prompt = f"""{system_prompt}

{format_prompt_with_instructions("What do you see in this image?")}"""
        
        return formatted_prompt
    
    def _build_context_aware_prompt(self, context_prompt: Dict, vision_results: Dict[str, Any], user_query: str) -> str:
        """
        Build context-aware prompt for FLAN-T5-Small with strong role priming
        Uses system prompt consistently and wraps user messages in instructions
        """
        # Strong role priming
        system_prompt = (
            "You are MonoVision AI, a conversational assistant. "
            "Answer in natural sentences. Do not be robotic or vague. "
            "Always provide clear, detailed, and human-like answers."
        )
        
        # Format user message with instructions wrapper
        def format_prompt_with_instructions(user_message: str) -> str:
            return f"You are a helpful AI assistant. Respond conversationally, be clear and complete.\n\nUser: {user_message}\nAssistant:"
        
        # Extract essential components with role context
        caption = vision_results.get("caption", "")
        
        # Build format with system priming
        if user_query and caption:
            user_message = f"{user_query} (Looking at image: {caption})"
            formatted_prompt = f"""{system_prompt}

{format_prompt_with_instructions(user_message)}"""
        elif user_query:
            # Text-only mode - provide conversational response without image context
            formatted_prompt = f"""{system_prompt}

{format_prompt_with_instructions(user_query)}"""
        elif caption:
            formatted_prompt = f"""{system_prompt}

{format_prompt_with_instructions(f"Describe this image: {caption}")}"""
        else:
            # Pure text-only fallback - be helpful and conversational
            formatted_prompt = f"""{system_prompt}

{format_prompt_with_instructions("Hello! How can I help you today?")}"""
        
        return formatted_prompt
    
    def _clean_response(self, response_text: str, original_prompt: str) -> str:
        """
        Enhanced response cleaning with strict code artifact removal for FLAN-T5-Small
        """
        # Remove the original prompt if it's included
        if original_prompt in response_text:
            response_text = response_text.replace(original_prompt, "").strip()
        
        # HARD BLOCK: Drop any text that looks like code (common FLAN-T5 artifact)
        code_indicators = ["def ", "import ", "class ", "from ", "print(", "return ", "if __name__", 
                          "```", "<code>", "</code>", ">>>", "...", ">>", "```python"]
        
        for indicator in code_indicators:
            if indicator in response_text:
                # Keep only the first line if code is detected
                lines = response_text.split('\n')
                for line in lines:
                    if not any(code_ind in line for code_ind in code_indicators):
                        response_text = line.strip()
                        break
                else:
                    # If all lines have code, take first few words only
                    words = response_text.split()[:5]
                    response_text = ' '.join(words)
                break
        
        # Remove common FLAN-T5 artifacts and system prompts
        response_text = response_text.strip()
        
        # Remove system prompts that may have leaked through
        system_leak_patterns = [
            "You are MonoVision AI", "You are a helpful AI assistant", "You are a conversational assistant",
            "Respond conversationally", "Be clear and complete", "Provide a clear, detailed response:",
            "Answer naturally:", "Answer:", "Question:", "Image:", "Describe:", "Description:"
        ]
        
        for pattern in system_leak_patterns:
            if response_text.startswith(pattern):
                response_text = response_text[len(pattern):].strip()
                # Remove trailing punctuation from system prompts
                while response_text.startswith((".", ":", ",")):
                    response_text = response_text[1:].strip()
        
        # Remove repetitive patterns
        lines = response_text.split('\n')
        cleaned_lines = []
        prev_line = ""
        
        for line in lines:
            line = line.strip()
            if line and line != prev_line and not any(pattern in line for pattern in system_leak_patterns):
                cleaned_lines.append(line)
                prev_line = line
        
        response_text = '\n'.join(cleaned_lines)
        
        # Remove common prefixes that FLAN-T5 might add
        prefixes_to_remove = [
            "Assistant:", "AI:", "Response:", "Answer:", "Output:", 
            "The image", "This image", "I can see", "Looking at",
            "Based on", "According to"
        ]
        
        for prefix in prefixes_to_remove:
            if response_text.startswith(prefix):
                response_text = response_text[len(prefix):].strip()
                if response_text.startswith(":"):
                    response_text = response_text[1:].strip()
        
        # Remove Google/Bing/Search artifacts that sometimes appear
        artifact_patterns = [
            "Google maps", "Bing maps", "Map showing", "Google search",
            "If you find something", "Please take a look", "Search results",
            "Wikipedia", "Stack Overflow"
        ]
        
        for pattern in artifact_patterns:
            if pattern.lower() in response_text.lower():
                # Try to salvage the useful part
                sentences = response_text.split('.')
                clean_sentences = [s.strip() for s in sentences 
                                 if not any(p.lower() in s.lower() for p in artifact_patterns)]
                if clean_sentences:
                    response_text = '. '.join(clean_sentences)
                    if not response_text.endswith('.'):
                        response_text += '.'
                else:
                    # If everything is corrupted, use fallback
                    response_text = ""
        
        # Remove HTML/XML tags that sometimes leak
        import re
        response_text = re.sub(r'<[^>]+>', '', response_text)
        
        # Remove special tokens that might leak
        special_tokens = ["<s>", "</s>", "<pad>", "<unk>", "<eos>", "<bos>"]
        for token in special_tokens:
            response_text = response_text.replace(token, "")
        
        # Ensure proper sentence structure
        response_text = response_text.strip()
        if response_text and not response_text.endswith(('.', '!', '?')):
            # Find the last complete sentence
            last_period = response_text.rfind('.')
            last_exclamation = response_text.rfind('!')
            last_question = response_text.rfind('?')
            
            last_punctuation = max(last_period, last_exclamation, last_question)
            
            if last_punctuation > len(response_text) * 0.6:  # If punctuation is in last 40%
                response_text = response_text[:last_punctuation + 1]
            else:
                response_text += "."
        
        return response_text
    
    def _enhance_weak_response(self, response_text: str, vision_results: Dict[str, Any], user_query: str) -> str:
        """
        Enhanced weak response detection and improvement with stronger fallbacks
        """
        # Check for specific weak patterns first (HIGHEST PRIORITY - more aggressive)
        weak_patterns = ["unclear", "unknown", "unsure", "blurry", "dark", "nothing", "empty", "something", "general", "basic"]
        
        response_lower = response_text.lower().strip()
        
        # Be more aggressive - check if ANY weak pattern exists ANYWHERE in response
        contains_weak_pattern = any(pattern in response_lower for pattern in weak_patterns)
        
        # Also check for overly generic responses that don't add value
        generic_responses = [
            "i can see", "this shows", "the image shows", "there is", "there are",
            "this is an image", "this is a picture", "this appears to be"
        ]
        
        is_too_generic = any(response_lower.startswith(pattern) for pattern in generic_responses) and len(response_text.split()) < 8
        
        if contains_weak_pattern or is_too_generic:
            if contains_weak_pattern:
                logger.info(f"🔍 Weak pattern detected in: '{response_text}' - applying fallback")
            else:
                logger.info(f"🔍 Generic response detected: '{response_text}' - applying enhancement")
            
            caption = vision_results.get("caption", "")
            keywords = vision_results.get("keywords", [])
            clip_keywords = vision_results.get("clip_keywords", [])
            all_keywords = keywords + clip_keywords
            
            # Don't use caption if it also contains weak patterns
            caption_is_weak = caption and any(pattern in caption.lower() for pattern in weak_patterns)
            
            if caption and not caption_is_weak:
                # Generate enhanced description from clean caption with more variety
                if "person" in caption.lower() or "people" in caption.lower():
                    responses = [
                        "I can see people in this image.",
                        "There are people visible in the scene.",
                        "The image shows people in what appears to be a social setting."
                    ]
                elif any(animal in caption.lower() for animal in ["animal", "cat", "dog", "bird", "horse", "pet"]):
                    responses = [
                        "There appears to be an animal in the image.",
                        "I can see what looks like a pet or animal in the scene.",
                        "The image features an animal as the main subject."
                    ]
                elif any(building in caption.lower() for building in ["building", "house", "structure", "architecture"]):
                    responses = [
                        "The image shows some architectural elements.",
                        "I can see a building or structure in the scene.",
                        "This appears to be an architectural photograph."
                    ]
                elif any(vehicle in caption.lower() for vehicle in ["car", "vehicle", "truck", "bus", "bike", "transport"]):
                    responses = [
                        "I can make out a vehicle in the scene.",
                        "The image shows some form of transportation.",
                        "There's a vehicle visible in this image."
                    ]
                elif any(food in caption.lower() for food in ["food", "meal", "eat", "kitchen", "cook", "restaurant"]):
                    responses = [
                        "This looks like it's related to food or dining.",
                        "I can see what appears to be a culinary scene.",
                        "The image shows something food-related."
                    ]
                elif any(nature in caption.lower() for nature in ["tree", "flower", "garden", "nature", "outdoor", "landscape"]):
                    responses = [
                        "This appears to be an outdoor or nature scene.",
                        "I can see natural elements in the image.",
                        "The image shows what looks like a natural setting."
                    ]
                else:
                    # More varied generic responses based on caption
                    responses = [
                        f"Looking at this image, I can see {caption.lower()}.",
                        f"The image captures {caption.lower()}.",
                        f"This shows {caption.lower()}, which is quite interesting.",
                        f"I notice {caption.lower()} in this scene."
                    ]
                
                import random
                return random.choice(responses) if isinstance(responses, list) else responses[0]
                
            elif all_keywords and not any(pattern in ' '.join(all_keywords).lower() for pattern in weak_patterns):
                # Use keywords if they're clean
                clean_keywords = [kw for kw in all_keywords[:3] if not any(wp in kw.lower() for wp in weak_patterns)]
                if clean_keywords:
                    keyword_responses = [
                        f"I can see elements related to {', '.join(clean_keywords)}.",
                        f"The image contains themes of {', '.join(clean_keywords)}.",
                        f"This appears to involve {', '.join(clean_keywords)}."
                    ]
                    import random
                    return random.choice(keyword_responses)
            
            # Use more varied generic fallbacks
            fallback_messages = [
                "I can see visual content in this image that I can analyze and discuss with you.",
                "The image contains identifiable elements that I can examine.",
                "There are distinguishable features visible in the scene that I can describe.",
                "I can make out several details in this image that might interest you.",
                "The image shows content that I can examine and provide insights about.",
                "This appears to be an interesting image with various elements to explore."
            ]
            import random
            return random.choice(fallback_messages)
        
        # Enforce minimum word count (critical for FLAN-T5-Small)
        word_count = len(response_text.split())
        
        if word_count < 6:  # Increased from 5 to 6 words minimum
            logger.info(f"🔍 Response too short ({word_count} words): '{response_text}' - applying fallback")
            
            # Generate a proper fallback response using vision context
            caption = vision_results.get("caption", "")
            keywords = vision_results.get("keywords", [])
            
            if caption:
                # Create more natural fallback responses
                enhanced_responses = [
                    f"This image shows {caption.lower()}.",
                    f"Looking at this, I can see {caption.lower()}.",
                    f"The scene depicts {caption.lower()}."
                ]
                import random
                return random.choice(enhanced_responses)
            elif keywords:
                enhanced_response = f"I can see elements related to {', '.join(keywords[:3])}."
                return enhanced_response
            else:
                return "I can see the image content and analyze what's shown."
        
        # Check for very generic single-word responses or overly simple phrases
        simple_patterns = [
            "yes", "no", "maybe", "image", "picture", "photo", "scene", "view", "this is", "that is"
        ]
        
        if response_text.lower().strip() in simple_patterns or any(response_text.lower().startswith(pattern) for pattern in simple_patterns[:6]):
            logger.info(f"🔍 Simple response detected: '{response_text}' - applying enhancement")
            caption = vision_results.get("caption", "")
            if caption:
                return f"The image shows {caption.lower()}."
            else:
                return "I can analyze and describe the visual content in this image."
        
        return response_text
        
        # Check for system leak artifacts (common with FLAN-T5-Small)
        system_artifacts = [
            "you are monovision", "you are a helpful", "conversational assistant",
            "respond conversationally", "be clear and complete", "provide a clear",
            "answer naturally", "question:", "image:", "describe:"
        ]
        
        if any(artifact in response_text.lower() for artifact in system_artifacts):
            logger.info(f"🔍 System artifact detected: '{response_text[:30]}...' - applying fallback")
            # Generate clean fallback from vision context
            caption = vision_results.get("caption", "")
            keywords = vision_results.get("keywords", [])
            
            if caption:
                fallback = f"Looking at this image, I can see {caption.lower()}."
                if user_query and user_query.strip():
                    fallback += f" Regarding your question, this captures what's visible in the scene."
                return fallback
            elif keywords:
                fallback = f"This image contains visual elements related to {', '.join(keywords[:3])}."
                return fallback
            else:
                return "I can analyze the image and provide information about what's shown."
        
        # Check for search engine artifacts (FLAN-T5 sometimes hallucinates these)
        search_artifacts = [
            "google maps", "bing maps", "search results", "wikipedia", "stack overflow",
            "if you find something", "please take a look", "click here", "website"
        ]
        
        if any(artifact in response_text.lower() for artifact in search_artifacts):
            logger.info(f"🔍 Search artifact detected: '{response_text[:30]}...' - applying fallback")
            caption = vision_results.get("caption", "")
            if caption:
                return f"The image shows {caption.lower()}."
            else:
                return "I can see the image and analyze its visual content."
        
        # Check for code artifacts (sometimes FLAN-T5 generates code)
        code_artifacts = [
            "def ", "import ", "class ", "print(", "return ", "if __name__",
            "```", "<code>", "python", "function"
        ]
        
        if any(artifact in response_text.lower() for artifact in code_artifacts):
            logger.info(f"🔍 Code artifact detected: '{response_text[:30]}...' - applying fallback")
            caption = vision_results.get("caption", "")
            if caption:
                return f"This image contains {caption.lower()}."
            else:
                return "I can see and analyze the image content."
        
        # Check for extremely short responses that might be incomplete
        if len(response_text.strip()) < 10:  # Less than 10 characters
            logger.info(f"🔍 Very short response detected: '{response_text}' - applying fallback")
            caption = vision_results.get("caption", "")
            if caption:
                return f"I can see {caption.lower()}."
            else:
                return "I can analyze what's shown in this image."
        
        return response_text
    
    def _vary_response_phrasing(self, response_text: str) -> str:
        """
        Add natural phrasing variation to avoid robotic repetition
        """
        import random
        
        # Phrasing starters to create variety
        starters = [
            "This image shows",
            "You can see", 
            "It looks like",
            "The picture contains",
            "I can see",
            "The image depicts",
            "Here we have",
            "This appears to be"
        ]
        
        # More aggressive phrasing variation
        response_lower = response_text.lower()
        
        # Replace repetitive "This image shows" with varied starters
        if response_text.startswith("This image shows"):
            starter = random.choice(starters)
            response_text = response_text.replace("This image shows", starter, 1)
        
        # Also handle "The image shows" pattern
        elif response_text.startswith("The image shows"):
            starter = random.choice(starters)
            response_text = response_text.replace("The image shows", starter, 1)
        
        # Handle "I can see" at the beginning
        elif response_text.startswith("I can see") and random.random() < 0.5:  # 50% chance to vary
            alternate_starters = ["You can see", "This shows", "The image contains", "Here we have"]
            starter = random.choice(alternate_starters)
            response_text = response_text.replace("I can see", starter, 1)
        
        # Handle "A [object] is [action]" pattern for more variety
        elif response_text.startswith("A ") and " is " in response_text and random.random() < 0.4:  # 40% chance
            # Transform "A dog is playing" -> "You can see a dog playing" / "Here's a dog playing"
            alternatives = [
                f"You can see {response_text[2:].replace(' is ', ' ')}", 
                f"Here's {response_text[2:].replace(' is ', ' ')}",
                f"The image shows {response_text[2:].replace(' is ', ' ')}",
                f"It looks like {response_text[2:].replace(' is ', ' ')}"
            ]
            response_text = random.choice(alternatives)
        
        # Handle greedy decoding repetition - since greedy always gives same output,
        # we need to add variation post-generation
        elif random.random() < 0.3:  # 30% chance to add variety even to other patterns
            # Add some variety to any response
            if response_text.startswith("There "):
                alternatives = ["You can see", "I notice", "The image contains"]
                replacement = random.choice(alternatives)
                response_text = response_text.replace("There ", f"{replacement} ", 1)
            
            elif response_text.startswith("It "):
                if random.random() < 0.5:
                    alternatives = ["This", "The image", "You can see"]
                    replacement = random.choice(alternatives)
                    response_text = response_text.replace("It ", f"{replacement} ", 1)
        
        return response_text
    
    def recently_used(self, threshold_minutes: int = 10) -> bool:
        """Check if model was used recently"""
        minutes_since_last_use = (time.time() - self.last_used) / 60
        return minutes_since_last_use < threshold_minutes
    
    async def unload(self):
        """Unload Flan-T5 model to free memory"""
        logger.info("🗑️ Unloading Flan-T5 model")
        
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        # Unregister memory allocation
        self.resource_manager.unregister_model_allocation("Flan-T5")
        
        # Clear GPU cache
        await self.resource_manager.cleanup_gpu_memory()
        
        logger.info("✅ Flan-T5 unloaded successfully")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Flan-T5 processing statistics"""
        avg_generation_time = (
            self.total_generation_time / self.generation_count 
            if self.generation_count > 0 else 0
        )
        
        return {
            "tier": "Tier 1 (Fast)",
            "model_name": self.model_name,
            "generation_count": self.generation_count,
            "total_generation_time": self.total_generation_time,
            "average_generation_time": avg_generation_time,
            "max_tokens": self.max_tokens,
            "device": self.device,
            "fp16_enabled": self.use_fp16,
            "loaded": self.model is not None,
            "last_used": self.last_used,
            "recently_used": self.recently_used()
        }

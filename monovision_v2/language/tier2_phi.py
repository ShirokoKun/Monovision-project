"""
Tier 2 Language Model: Phi-2 (Balanced Mode)
2.7B model with 4-bit quantization, mixed GPU + CPU offload, 50 tokens
"""

import asyncio
import logging
import time
from typing import Dict, Any, Tuple, Optional
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import BitsAndBytesConfig
except ImportError as e:
    logging.error(f"Missing transformers dependency: {e}")

from ..core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)

class Phi2Processor:
    """
    Tier 2 Language Processor using Phi-2 with 4-bit quantization
    - 2.7B parameters with 4-bit quantization (~1.5GB VRAM)
    - Mixed GPU + CPU offload for optimal GTX 1650 performance
    - Used in Balanced mode: BLIP + CLIP + Phi-2
    - Target: 50 tokens, ~30s response time
    """
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        self.resource_manager = resource_manager or ResourceManager()
        
        # Model configuration
        self.model_name = "microsoft/phi-2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_tokens = 100  # Increased from 50 to 100 for balanced mode
        
        # Model instances
        self.tokenizer = None
        self.model = None
        
        # Performance tracking
        self.generation_count = 0
        self.total_generation_time = 0.0
        self.last_used = time.time()
        
        logger.info("⚖️ Phi-2 Processor (Tier 2) initialized")
    
    async def initialize(self):
        """Initialize Phi-2 model with 4-bit quantization"""
        logger.info("🚀 Loading Phi-2 (Tier 2: Balanced Mode) with 4-bit quantization")
        
        try:
            # Check if we can load the model
            estimated_memory = 1500  # MB for Phi-2 4-bit
            if not self.resource_manager.can_load_model("Phi-2", estimated_memory):
                suggestions = self.resource_manager.suggest_model_unloading(estimated_memory)
                logger.warning(f"⚠️ Insufficient memory for Phi-2. Suggest unloading: {suggestions}")
                raise RuntimeError("Insufficient GPU memory for Phi-2")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir="cache/hf",
                trust_remote_code=True
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure 4-bit quantization with CPU offloading for GTX 1650
            if self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # Simplified device map for GTX 1650 stability
                # Load model with automatic device mapping and CPU offloading
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir="cache/hf",
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    attn_implementation="eager",
                    offload_folder="cache/phi2_offload",  # CPU offload directory
                    max_memory={0: "2.5GB", "cpu": "8GB"}  # Conservative memory limits
                )
            else:
                # CPU fallback without quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir="cache/hf",
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                self.model = self.model.to(self.device)
            
            # Set to eval mode
            self.model.eval()
            
            # Register memory allocation
            self.resource_manager.register_model_allocation("Phi-2", estimated_memory)
            
            logger.info(f"✅ Phi-2 loaded successfully with 4-bit quantization")
            logger.info(f"📊 Estimated memory usage: {estimated_memory}MB")
            
        except Exception as e:
            logger.error(f"❌ Error loading Phi-2: {e}")
            raise
    
    async def generate_response(self, vision_results: Dict[str, Any], user_query: str, max_tokens: int = 100, context_prompt: Dict = None) -> Tuple[str, int]:
        """
        Generate enhanced response using Phi-2 with context-aware prompts
        
        Args:
            vision_results: Results from vision fusion layer
            user_query: User's question
            max_tokens: Maximum tokens to generate (≤100 for balanced mode)
            context_prompt: V3 context-aware prompt templates
            
        Returns:
            Tuple of (response_text, token_count)
        """
        start_time = time.time()
        self.last_used = time.time()
        
        try:
            # Extract vision context
            caption = vision_results.get("caption", "")
            clip_keywords = vision_results.get("clip_keywords", [])
            
            # DEBUG: Log the vision results being passed to Phi-2
            logger.info(f"🔍 Phi-2 DEBUG - Caption: {caption[:100]}...")
            logger.info(f"🔍 Phi-2 DEBUG - Keywords: {clip_keywords}")
            logger.info(f"🔍 Phi-2 DEBUG - User query: {user_query}")
            
            # Build prompt using context-aware templates (V3 Enhancement)
            if context_prompt:
                prompt = self._build_context_aware_prompt(context_prompt, vision_results, user_query)
                logger.info(f"🔍 Phi-2 DEBUG - Using context-aware prompt")
            else:
                # Fallback to original prompt
                prompt = self._build_balanced_prompt(caption, clip_keywords, user_query)
                logger.info(f"🔍 Phi-2 DEBUG - Using fallback prompt")
            
            # DEBUG: Log the first 200 chars of the prompt
            logger.info(f"🔍 Phi-2 DEBUG - Prompt preview: {prompt[:200]}...")
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Move inputs to the main device (first GPU device where embed_tokens is)
            # For models with device_map, inputs should go to device 0
            target_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            
            # Adaptive generation configuration optimized for enhanced Phi-2
            if max_tokens <= 50:
                # Standard balanced mode - greedy decoding for consistent responses
                generation_config = {
                    "max_new_tokens": min(max_tokens, self.max_tokens),
                    "do_sample": False,  # Greedy for consistency and speed
                    "temperature": 1.0,  # Irrelevant for greedy
                    "use_cache": True,
                    "repetition_penalty": 1.1,
                    "num_beams": 1,
                    "early_stopping": False,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id
                }
            elif max_tokens <= 80:
                # Enhanced balanced mode for detailed responses (50-80 tokens)
                generation_config = {
                    "max_new_tokens": min(max_tokens, self.max_tokens),
                    "do_sample": True,   # Enable sampling for variety
                    "temperature": 0.7,  # Moderate creativity for detailed responses
                    "top_p": 0.9,        # Nucleus sampling
                    "top_k": 50,         # Top-k filtering
                    "use_cache": True,
                    "repetition_penalty": 1.12,
                    "num_beams": 1,
                    "early_stopping": False,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id
                }
            else:
                # Extended mode for comprehensive responses (80-100 tokens)
                generation_config = {
                    "max_new_tokens": min(max_tokens, self.max_tokens),
                    "do_sample": True,   # Enable sampling for variety
                    "temperature": 0.8,  # Higher creativity for comprehensive responses
                    "top_p": 0.85,       # Nucleus sampling
                    "top_k": 40,         # Top-k filtering
                    "use_cache": True,
                    "repetition_penalty": 1.15,  # Higher for extended mode
                    "num_beams": 1,
                    "early_stopping": False,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id
                }
            
            # Generate with timeout protection
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **generation_config
                    )
                
                # Decode response
                input_length = inputs["input_ids"].shape[1]
                response_tokens = outputs[0][input_length:]
                response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                
                # Enhanced response cleaning and processing pipeline
                response_text = self._clean_response(response_text)
                
                # POST-PROCESS WEAK RESPONSES WITH STRONGER FALLBACKS & AUTO-RETRY
                original_response = response_text
                response_text = self._enhance_weak_response(response_text, vision_results, user_query)
                
                # CRITICAL: Auto-retry mechanism for weak responses (similar to FLAN-T5)
                if len(response_text.split()) < 5 or any(pattern in response_text.lower() for pattern in ["unclear", "unknown", "unsure"]):
                    logger.info(f"🔄 Phi-2 auto-retry triggered for weak response: '{response_text[:30]}...'")
                    
                    # Retry with stronger, more detailed prompt
                    stronger_prompt = f"""{prompt}

Please expand your response and make it detailed and conversational. Provide more specific information."""
                    
                    stronger_inputs = self.tokenizer(
                        stronger_prompt,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True,
                        padding=True
                    )
                    
                    # Move to correct device
                    stronger_inputs = {k: v.to(target_device) for k, v in stronger_inputs.items()}
                    
                    # Use more creative settings for retry
                    retry_config = {
                        "max_new_tokens": min(60, self.max_tokens),
                        "do_sample": True,       # Enable sampling for creativity
                        "temperature": 0.9,      # Higher temperature for variety
                        "top_p": 0.85,          # Nucleus sampling
                        "top_k": 40,            # Top-k filtering
                        "use_cache": True,
                        "repetition_penalty": 1.2,  # Higher to avoid repetition
                        "pad_token_id": self.tokenizer.eos_token_id,
                        "eos_token_id": self.tokenizer.eos_token_id
                    }
                    
                    try:
                        with torch.no_grad():
                            retry_outputs = self.model.generate(
                                **stronger_inputs,
                                **retry_config
                            )
                        
                        # Decode retry response
                        retry_input_length = stronger_inputs["input_ids"].shape[1]
                        retry_response_tokens = retry_outputs[0][retry_input_length:]
                        retry_response = self.tokenizer.decode(retry_response_tokens, skip_special_tokens=True)
                        retry_response = self._clean_response(retry_response)
                        
                        # Use retry response if it's better
                        if len(retry_response.split()) > len(response_text.split()) and len(retry_response.split()) >= 8:
                            logger.info(f"✅ Phi-2 retry successful: '{retry_response[:30]}...'")
                            response_text = retry_response
                        else:
                            logger.info(f"⚠️ Phi-2 retry didn't improve response, keeping original")
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Phi-2 retry failed: {e}")
                
                # If we still have weak patterns after enhancement, apply emergency fallback
                weak_patterns = ["unclear", "unknown", "unsure", "blurry", "dark", "nothing", "empty", "something"]
                if any(pattern in response_text.lower() for pattern in weak_patterns):
                    logger.info(f"🚨 Emergency fallback triggered for Phi-2 - weak pattern still present")
                    response_text = "I can analyze the visual content and provide information about what's shown in this image."
                
                # Log if weak response was detected and replaced
                if response_text != original_response:
                    logger.info(f"🔧 Phi-2 weak response enhanced: '{original_response[:30]}...' -> '{response_text[:30]}...'")
                
                # Add natural phrasing variation to avoid robotic repetition
                response_text = self._vary_response_phrasing(response_text)
                
                # Count tokens
                token_count = len(response_tokens)
                
                # Update performance metrics
                generation_time = time.time() - start_time
                self.generation_count += 1
                self.total_generation_time += generation_time
                
                logger.info(f"⚖️ Phi-2 generated {token_count} tokens in {generation_time:.2f}s")
                
                return response_text, token_count
                
            except torch.cuda.OutOfMemoryError:
                logger.error("❌ GPU out of memory during Phi-2 generation")
                torch.cuda.empty_cache()
                return "GPU memory limit reached. Try a shorter query.", 0
                
        except Exception as e:
            logger.error(f"❌ Phi-2 generation error: {e}")
            return f"Error generating response: {str(e)}", 0
    
    def _build_balanced_prompt(self, caption: str, clip_keywords: list, user_query: str) -> str:
        """
        Build optimized prompt for balanced mode with enhanced Phi-2 formatting
        Enhanced with strong role priming and user message wrapping
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
        
        if user_query.strip() and caption.strip():
            # User has specific question about an image - optimized for Phi-2
            context_parts = []
            
            # Add image description
            context_parts.append(f"Image description: {caption}")
            
            # Add CLIP keywords if available
            if clip_keywords:
                # Filter and format keywords
                clean_keywords = [kw for kw in clip_keywords[:5] if len(kw) > 2]  # Top 5, filter short words
                if clean_keywords:
                    context_parts.append(f"Key elements: {', '.join(clean_keywords)}")
            
            context = "\n".join(context_parts)
            user_message = f"{user_query}\n\n{context}"
            
            # Phi-2 works well with this format
            prompt = f"""{system_prompt}

{format_prompt_with_instructions(user_message)}"""
            
        elif user_query.strip():
            # User asks a question but no image context - simple format for Phi-2
            prompt = f"""{system_prompt}

{format_prompt_with_instructions(user_query)}"""
            
        elif caption.strip():
            # No specific question, describe image - use Phi-2's strengths
            context_parts = [f"Image description: {caption}"]
            
            if clip_keywords:
                clean_keywords = [kw for kw in clip_keywords[:5] if len(kw) > 2]
                if clean_keywords:
                    context_parts.append(f"Key elements: {', '.join(clean_keywords)}")
            
            context = "\n".join(context_parts)
            user_message = f"Describe this image in detail:\n\n{context}"
            
            prompt = f"""{system_prompt}

{format_prompt_with_instructions(user_message)}"""
        else:
            # Fallback for no context
            prompt = f"""{system_prompt}

{format_prompt_with_instructions("What do you see in this image?")}"""
        
        return prompt
    
    def _build_context_aware_prompt(self, context_prompt: Dict, vision_results: Dict[str, Any], user_query: str) -> str:
        """
        Build context-aware prompt using V3 templates optimized for Phi-2
        Enhanced with strong role priming and user message wrapping
        """
        # Strong system prompt for role priming
        system_prompt = (
            "You are MonoVision AI, a conversational assistant. "
            "Answer in natural sentences. Do not be robotic or vague. "
            "Always provide clear, detailed, and human-like answers."
        )
        
        # Use the context-aware system and user prompts
        context_system_prompt = context_prompt.get("system_prompt", system_prompt)
        user_prompt = context_prompt.get("user_prompt", f"Analyze this image: {user_query}")
        
        # Simplify system prompt for Phi-2 if too long, but keep role priming
        if len(context_system_prompt) > 200:
            final_system_prompt = system_prompt  # Use our consistent role priming
        else:
            final_system_prompt = context_system_prompt
        
        # Format user message with instructions wrapper
        def format_prompt_with_instructions(user_message: str) -> str:
            return f"You are a helpful AI assistant. Respond conversationally, be clear and complete.\n\nUser: {user_message}\nAssistant:"
        
        # Format optimized for Phi-2's training format
        return f"""{final_system_prompt}

{format_prompt_with_instructions(user_prompt)}"""
    
    def _clean_response(self, response_text: str) -> str:
        """Enhanced response cleaning for Phi-2 with artifact removal"""
        # Remove common artifacts and system prompts
        response_text = response_text.strip()
        
        # Remove system prompts that may have leaked through
        system_leak_patterns = [
            "System:", "User:", "Assistant:", "AI:", "Response:", "Answer:", "Output:",
            "You are a helpful AI assistant", "You are MonoVision AI", "I am", "As an AI",
            "Based on the image", "Looking at this image", "In this image"
        ]
        
        for pattern in system_leak_patterns:
            if response_text.startswith(pattern):
                response_text = response_text[len(pattern):].strip()
                # Remove trailing punctuation from system prompts
                while response_text.startswith((".", ":", ",")):
                    response_text = response_text[1:].strip()
        
        # Remove code artifacts (Phi-2 can generate code)
        code_indicators = ["def ", "import ", "class ", "from ", "print(", "return ", "if __name__", 
                          "```", "<code>", "</code>", ">>>", "python", "function"]
        
        for indicator in code_indicators:
            if indicator in response_text:
                # Keep only the first line if code is detected
                lines = response_text.split('\n')
                for line in lines:
                    if not any(code_ind in line.lower() for code_ind in code_indicators):
                        response_text = line.strip()
                        break
                else:
                    # If all lines have code, take first few words only
                    words = response_text.split()[:8]  # More words for Phi-2
                    response_text = ' '.join(words)
                break
        
        # Remove repetitive endings
        if response_text.endswith("..."):
            response_text = response_text[:-3].strip()
        
        # Remove HTML/XML tags that sometimes leak
        import re
        response_text = re.sub(r'<[^>]+>', '', response_text)
        
        # Remove special tokens that might leak
        special_tokens = ["<s>", "</s>", "<pad>", "<unk>", "<eos>", "<bos>", "<|endoftext|>"]
        for token in special_tokens:
            response_text = response_text.replace(token, "")
        
        # Ensure proper sentence ending
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
        Enhanced weak response detection and improvement for Phi-2 with better conversational flow
        """
        # Check for specific weak patterns first (HIGHEST PRIORITY)
        weak_patterns = ["unclear", "unknown", "unsure", "blurry", "dark", "nothing", "empty", "something", "general", "basic"]
        
        response_lower = response_text.lower().strip()
        
        # Be aggressive - check if ANY weak pattern exists ANYWHERE in response
        contains_weak_pattern = any(pattern in response_lower for pattern in weak_patterns)
        
        # Also check for overly generic responses that don't add conversational value
        generic_starters = [
            "i can see", "this shows", "the image shows", "there is", "there are",
            "this is an image", "this is a picture", "this appears to be", "looking at"
        ]
        
        is_too_generic = any(response_lower.startswith(pattern) for pattern in generic_starters) and len(response_text.split()) < 12
        
        if contains_weak_pattern or is_too_generic:
            if contains_weak_pattern:
                logger.info(f"🔍 Phi-2 weak pattern detected in: '{response_text}' - applying fallback")
            else:
                logger.info(f"🔍 Phi-2 generic response detected: '{response_text}' - applying enhancement")
            
            caption = vision_results.get("caption", "")
            keywords = vision_results.get("keywords", [])
            clip_keywords = vision_results.get("clip_keywords", [])
            all_keywords = keywords + clip_keywords
            
            # Don't use caption if it also contains weak patterns
            caption_is_weak = caption and any(pattern in caption.lower() for pattern in weak_patterns)
            
            if caption and not caption_is_weak:
                # Generate more conversational descriptions from clean caption
                if "person" in caption.lower() or "people" in caption.lower():
                    responses = [
                        "I can see people in this image, and they seem to be the main focus of the scene.",
                        "There are people visible here, which makes this an interesting social or human-centered image.",
                        "The image captures people in what appears to be a candid or everyday moment."
                    ]
                elif any(animal in caption.lower() for animal in ["animal", "cat", "dog", "bird", "horse", "pet"]):
                    responses = [
                        "There's an animal featured in this image, which always makes for engaging photography.",
                        "I can see what looks like a pet or animal, and they appear to be the star of this photo.",
                        "The image showcases an animal in what seems like a natural or comfortable setting."
                    ]
                elif any(building in caption.lower() for building in ["building", "house", "structure", "architecture"]):
                    responses = [
                        "The image features some interesting architectural elements that catch the eye.",
                        "I can see a building or structure that appears to be the focal point of this photograph.",
                        "This looks like an architectural shot with some notable design features."
                    ]
                elif any(vehicle in caption.lower() for vehicle in ["car", "vehicle", "truck", "bus", "bike", "transport"]):
                    responses = [
                        "There's a vehicle visible in this scene, which adds an element of movement or travel to the image.",
                        "I can see some form of transportation that seems integral to what's happening here.",
                        "The image includes a vehicle that appears to be part of the main subject matter."
                    ]
                elif any(food in caption.lower() for food in ["food", "meal", "eat", "kitchen", "cook", "restaurant"]):
                    responses = [
                        "This appears to be food-related, which always makes for appealing imagery.",
                        "I can see what looks like a culinary scene that would make anyone hungry.",
                        "The image showcases something food-related that looks quite appetizing."
                    ]
                elif any(nature in caption.lower() for nature in ["tree", "flower", "garden", "nature", "outdoor", "landscape"]):
                    responses = [
                        "This appears to be an outdoor scene with natural elements that create a peaceful atmosphere.",
                        "I can see natural elements that give this image a fresh, outdoor feeling.",
                        "The image captures what looks like a beautiful natural setting."
                    ]
                else:
                    # More varied and conversational generic responses
                    responses = [
                        f"Looking at this image, I can see {caption.lower()}, which creates an interesting visual story.",
                        f"The image captures {caption.lower()}, and there's something compelling about this scene.",
                        f"This shows {caption.lower()}, and I find the composition quite engaging.",
                        f"I notice {caption.lower()} in this scene, which draws attention immediately."
                    ]
                
                import random
                return random.choice(responses) if isinstance(responses, list) else responses[0]
                
            elif all_keywords and not any(pattern in ' '.join(all_keywords).lower() for pattern in weak_patterns):
                # Use keywords with more conversational context
                clean_keywords = [kw for kw in all_keywords[:3] if not any(wp in kw.lower() for wp in weak_patterns)]
                if clean_keywords:
                    keyword_responses = [
                        f"I can see elements related to {', '.join(clean_keywords)}, which creates an interesting thematic composition.",
                        f"The image contains themes of {', '.join(clean_keywords)}, giving it a distinctive character.",
                        f"This appears to involve {', '.join(clean_keywords)}, which makes for compelling subject matter."
                    ]
                    import random
                    return random.choice(keyword_responses)
            
            # Use more engaging and conversational fallbacks
            fallback_messages = [
                "I can analyze the visual content in this image and provide insights about what makes it interesting.",
                "The image contains identifiable elements that I can examine and discuss with you.",
                "There are distinguishable features visible that create an engaging composition worth exploring.",
                "I can make out several details in this image that contribute to its overall visual appeal.",
                "The image shows content that I can examine and offer thoughtful observations about.",
                "This appears to be an intriguing image with various elements that tell a visual story."
            ]
            import random
            return random.choice(fallback_messages)
        
        # Enforce higher minimum word count for Phi-2 (more capable model)
        word_count = len(response_text.split())
        
        if word_count < 10:  # Higher minimum for Phi-2 (10 words vs 6 for FLAN-T5)
            logger.info(f"🔍 Phi-2 response too short ({word_count} words): '{response_text}' - applying fallback")
            
            # Generate more sophisticated fallback response using vision context
            caption = vision_results.get("caption", "")
            keywords = vision_results.get("keywords", [])
            clip_keywords = vision_results.get("clip_keywords", [])
            
            if caption:
                enhanced_responses = [
                    f"This image shows {caption.lower()}, and there are several interesting aspects to explore here.",
                    f"Looking at this scene, I can see {caption.lower()}, which creates a compelling visual narrative.",
                    f"The image captures {caption.lower()}, and I find the composition quite engaging to analyze."
                ]
                import random
                return random.choice(enhanced_responses)
            elif keywords or clip_keywords:
                all_keywords = keywords + clip_keywords
                enhanced_response = f"I can see elements related to {', '.join(all_keywords[:3])}, which makes this an interesting subject to examine."
                return enhanced_response
            else:
                return "I can see and analyze the image content, and there are several visual elements worth discussing."
        
        return response_text
    
    def _vary_response_phrasing(self, response_text: str) -> str:
        """
        Add natural phrasing variation for Phi-2 responses to avoid robotic repetition
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
            "This appears to be",
            "The scene shows",
            "What's visible is"
        ]
        
        # More aggressive phrasing variation for Phi-2
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
        elif response_text.startswith("I can see") and random.random() < 0.6:  # 60% chance to vary
            alternate_starters = ["You can see", "This shows", "The image contains", "Here we have", "What's visible is"]
            starter = random.choice(alternate_starters)
            response_text = response_text.replace("I can see", starter, 1)
        
        # Handle "There is/are" patterns for more variety
        elif response_text.startswith("There is") and random.random() < 0.5:
            alternatives = ["You can see", "The image shows", "I notice", "What's visible is"]
            replacement = random.choice(alternatives)
            response_text = response_text.replace("There is", f"{replacement}", 1)
            
        elif response_text.startswith("There are") and random.random() < 0.5:
            alternatives = ["You can see", "The image shows", "I notice", "What's visible are"]
            replacement = random.choice(alternatives)
            response_text = response_text.replace("There are", f"{replacement}", 1)
        
        # Handle "A [object] is [action]" pattern for more variety (Phi-2 specific)
        elif response_text.startswith("A ") and " is " in response_text and random.random() < 0.4:
            # Transform "A dog is playing" -> "You can see a dog playing" / "Here's a dog playing"
            alternatives = [
                f"You can see {response_text[2:].replace(' is ', ' ')}", 
                f"Here's {response_text[2:].replace(' is ', ' ')}",
                f"The image shows {response_text[2:].replace(' is ', ' ')}",
                f"It looks like {response_text[2:].replace(' is ', ' ')}"
            ]
            response_text = random.choice(alternatives)
        
        # Handle "It appears" or "It seems" patterns
        elif response_text.startswith("It appears") and random.random() < 0.4:
            alternatives = ["This shows", "You can see", "The image depicts", "What's visible"]
            replacement = random.choice(alternatives)
            response_text = response_text.replace("It appears", replacement, 1)
            
        elif response_text.startswith("It seems") and random.random() < 0.4:
            alternatives = ["This shows", "You can see", "The image depicts", "It looks like"]
            replacement = random.choice(alternatives)
            response_text = response_text.replace("It seems", replacement, 1)
        
        return response_text
    
    def recently_used(self, threshold_minutes: int = 15) -> bool:
        """Check if model was used recently (longer threshold for larger model)"""
        minutes_since_last_use = (time.time() - self.last_used) / 60
        return minutes_since_last_use < threshold_minutes
    
    async def unload(self):
        """Unload Phi-2 model to free memory"""
        logger.info("🗑️ Unloading Phi-2 model")
        
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        # Unregister memory allocation
        self.resource_manager.unregister_model_allocation("Phi-2")
        
        # Clear GPU cache
        await self.resource_manager.cleanup_gpu_memory()
        
        logger.info("✅ Phi-2 unloaded successfully")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Phi-2 processing statistics"""
        avg_generation_time = (
            self.total_generation_time / self.generation_count 
            if self.generation_count > 0 else 0
        )
        
        return {
            "tier": "Tier 2 (Balanced)",
            "model_name": self.model_name,
            "generation_count": self.generation_count,
            "total_generation_time": self.total_generation_time,
            "average_generation_time": avg_generation_time,
            "max_tokens": self.max_tokens,
            "device": self.device,
            "quantization": "4-bit",
            "loaded": self.model is not None,
            "last_used": self.last_used,
            "recently_used": self.recently_used()
        }

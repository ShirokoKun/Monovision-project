"""
Tier 3 Language Model: Mistral-7B (Rich Mode)
7B model with CPU offloading, 50-100 tokens, comprehensive analysis
"""

import asyncio
import logging
import time
from typing import Dict, Any, Tuple, Optional
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from accelerate import infer_auto_device_map, dispatch_model
except ImportError as e:
    logging.error(f"Missing transformers/accelerate dependency: {e}")

from ..core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)

class MistralProcessor:
    """
    Tier 3 Language Processor using Mistral-7B with CPU offloading
    - 7B parameters with strategic CPU offloading (60% CPU, 40% GPU)
    - Uses 16GB RAM for offloaded layers
    - Used in Rich mode: BLIP + CLIP + Mistral-7B
    - Target: 50-100 tokens, comprehensive analysis
    """
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        self.resource_manager = resource_manager or ResourceManager()
        
        # Model configuration
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_tokens = 100  # Rich mode limit
        self.offload_ratio = 0.6  # 60% to CPU, 40% on GPU
        
        # Model instances
        self.tokenizer = None
        self.model = None
        
        # Performance tracking
        self.generation_count = 0
        self.total_generation_time = 0.0
        self.last_used = time.time()
        
        logger.info("🎯 Mistral-7B Processor (Tier 3) initialized")
    
    async def initialize(self):
        """Initialize Mistral-7B model with CPU offloading"""
        logger.info("🚀 Loading Mistral-7B (Tier 3: Rich Mode) with CPU offloading")
        
        try:
            # Check system requirements
            cpu_info = self.resource_manager.get_cpu_usage()
            if not cpu_info.get("suitable_for_mistral", False):
                logger.warning("⚠️ System may not have sufficient RAM for Mistral-7B offloading")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir="cache/hf"
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            if self.device == "cuda":
                # Load model with CPU offloading strategy
                logger.info(f"📊 Configuring CPU offloading: {self.offload_ratio*100}% CPU, {(1-self.offload_ratio)*100}% GPU")
                
                # First, load model in CPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir="cache/hf",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="cpu"  # Start on CPU
                )
                
                # Create custom device map for offloading
                device_map = self._create_offload_device_map()
                
                # Dispatch model with custom device map
                self.model = dispatch_model(self.model, device_map=device_map)
                
                # Register memory allocation (GPU portion only)
                gpu_memory_estimate = 1600  # MB for ~40% of 7B model
                self.resource_manager.register_model_allocation("Mistral-7B", gpu_memory_estimate)
                
            else:
                # CPU-only fallback
                logger.info("💻 Loading Mistral-7B on CPU only")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir="cache/hf",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                self.model = self.model.to(self.device)
            
            # Set to eval mode
            self.model.eval()
            
            logger.info(f"✅ Mistral-7B loaded successfully with offloading")
            logger.info(f"📊 Device allocation: {getattr(self.model, 'hf_device_map', 'CPU-only')}")
            
        except Exception as e:
            logger.error(f"❌ Error loading Mistral-7B: {e}")
            raise
    
    def _create_offload_device_map(self) -> Dict[str, str]:
        """
        Create device map for strategic CPU offloading
        Places ~40% on GPU, ~60% on CPU based on layer importance
        """
        try:
            # Get model config to understand layer structure
            num_layers = self.model.config.num_hidden_layers
            
            # Calculate GPU vs CPU allocation
            gpu_layers = int(num_layers * (1 - self.offload_ratio))  # 40% on GPU
            
            device_map = {}
            
            # Embedding and final layers on GPU (most important)
            device_map["model.embed_tokens"] = 0
            device_map["model.norm"] = 0
            device_map["lm_head"] = 0
            
            # First few transformer layers on GPU (attention is GPU-optimized)
            for i in range(gpu_layers):
                device_map[f"model.layers.{i}"] = 0
            
            # Remaining layers on CPU
            for i in range(gpu_layers, num_layers):
                device_map[f"model.layers.{i}"] = "cpu"
            
            logger.info(f"📊 Device map created: {gpu_layers} layers on GPU, {num_layers - gpu_layers} on CPU")
            return device_map
            
        except Exception as e:
            logger.warning(f"⚠️ Error creating device map: {e}, using auto")
            return "auto"
    
    async def generate_response(self, vision_results: Dict[str, Any], user_query: str, max_tokens: int = 100, context_prompt: Dict = None) -> Tuple[str, int]:
        """
        Generate comprehensive response using Mistral-7B
        
        Args:
            vision_results: Results from vision fusion layer
            user_query: User's question
            max_tokens: Maximum tokens to generate (≤100 for rich mode)
            
        Returns:
            Tuple of (response_text, token_count)
        """
        start_time = time.time()
        self.last_used = time.time()
        
        try:
            # Extract comprehensive vision context
            caption = vision_results.get("caption", "")
            clip_keywords = vision_results.get("clip_keywords", [])
            objects = vision_results.get("objects", [])
            
            # Build prompt for rich mode
            prompt = self._build_rich_prompt(caption, clip_keywords, objects, user_query)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,  # Longer context for rich mode
                truncation=True,
                padding=True
            )
            
            # Move to appropriate device
            if not hasattr(self.model, 'hf_device_map'):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                # Model has device map, move to first device
                first_device = next(iter(set(self.model.hf_device_map.values())))
                if first_device != "cpu":
                    inputs = {k: v.to(first_device) for k, v in inputs.items()}
            
            # Generation configuration for comprehensive analysis
            generation_config = {
                "max_new_tokens": min(max_tokens, self.max_tokens),
                "do_sample": True,  # Sampling for more diverse outputs
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "use_cache": True,
                "early_stopping": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }
            
            # Generate with extended timeout for CPU offloading
            logger.info("🎯 Starting Mistral-7B generation (extended processing time expected)")
            
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
                
                # Clean up response
                response_text = self._clean_rich_response(response_text)
                
                # Count tokens
                token_count = len(response_tokens)
                
                # Update performance metrics
                generation_time = time.time() - start_time
                self.generation_count += 1
                self.total_generation_time += generation_time
                
                logger.info(f"🎯 Mistral-7B generated {token_count} tokens in {generation_time:.2f}s")
                
                return response_text, token_count
                
            except torch.cuda.OutOfMemoryError:
                logger.error("❌ GPU out of memory during Mistral-7B generation")
                torch.cuda.empty_cache()
                return "GPU memory limit reached during comprehensive analysis.", 0
                
        except Exception as e:
            logger.error(f"❌ Mistral-7B generation error: {e}")
            return f"Error generating comprehensive response: {str(e)}", 0
    
    def _build_rich_prompt(self, caption: str, clip_keywords: list, objects: list, user_query: str) -> str:
        """
        Build comprehensive prompt for rich mode analysis
        Include all available vision information
        """
        # Build detailed context
        context_parts = [
            f"Image Description: {caption}",
        ]
        
        if clip_keywords:
            # Handle both string and dict formats for clip_keywords
            if clip_keywords and isinstance(clip_keywords[0], dict):
                keyword_strings = [str(kw.get('name', kw.get('keyword', str(kw)))) for kw in clip_keywords]
            else:
                keyword_strings = [str(kw) for kw in clip_keywords]
            context_parts.append(f"Visual Style: {', '.join(keyword_strings)}")
        
        if objects:
            # Handle both string and dict formats for objects
            if objects and isinstance(objects[0], dict):
                object_strings = [str(obj.get('name', str(obj))) for obj in objects]
            else:
                object_strings = [str(obj) for obj in objects]
            context_parts.append(f"Detected Objects: {', '.join(object_strings)}")
        
        context = "\n".join(context_parts)
        
        if user_query.strip():
            # User has specific question - provide comprehensive analysis
            prompt = f"""<s>[INST] You are an expert image analyst. Based on the detailed image analysis below, provide a comprehensive and insightful answer to the user's question.

{context}

User Question: {user_query}

Provide a detailed, analytical response that considers visual elements, composition, context, and potential meanings. [/INST]

"""
        else:
            # No specific question - provide comprehensive description
            prompt = f"""<s>[INST] You are an expert image analyst. Based on the detailed image analysis below, provide a comprehensive description and analysis.

{context}

Provide a thorough analysis covering visual elements, composition, mood, context, and any interesting observations about this image. [/INST]

"""
        
        return prompt
    
    def _clean_rich_response(self, response_text: str) -> str:
        """Clean up the generated comprehensive response"""
        # Remove common artifacts
        response_text = response_text.strip()
        
        # Remove instruction tags if they appear
        if response_text.startswith("</s>"):
            response_text = response_text[4:].strip()
        
        # Ensure proper paragraph structure
        if len(response_text) > 200:  # Long response
            # Add paragraph breaks for readability
            sentences = response_text.split('. ')
            if len(sentences) > 3:
                # Group sentences into paragraphs
                paragraphs = []
                current_paragraph = []
                
                for i, sentence in enumerate(sentences):
                    current_paragraph.append(sentence)
                    
                    # Create paragraph break every 2-3 sentences
                    if (i + 1) % 3 == 0 and i > 0:
                        paragraphs.append('. '.join(current_paragraph) + '.')
                        current_paragraph = []
                
                # Add remaining sentences
                if current_paragraph:
                    paragraphs.append('. '.join(current_paragraph))
                
                response_text = '\n\n'.join(paragraphs)
        
        # Ensure proper ending
        if response_text and not response_text.endswith(('.', '!', '?')):
            response_text += "."
        
        return response_text
    
    def recently_used(self, threshold_minutes: int = 20) -> bool:
        """Check if model was used recently (longer threshold for heavy model)"""
        minutes_since_last_use = (time.time() - self.last_used) / 60
        return minutes_since_last_use < threshold_minutes
    
    async def unload(self):
        """Unload Mistral-7B model to free memory"""
        logger.info("🗑️ Unloading Mistral-7B model")
        
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        # Unregister memory allocation
        self.resource_manager.unregister_model_allocation("Mistral-7B")
        
        # Clear GPU cache
        await self.resource_manager.cleanup_gpu_memory()
        
        logger.info("✅ Mistral-7B unloaded successfully")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Mistral-7B processing statistics"""
        avg_generation_time = (
            self.total_generation_time / self.generation_count 
            if self.generation_count > 0 else 0
        )
        
        return {
            "tier": "Tier 3 (Rich)",
            "model_name": self.model_name,
            "generation_count": self.generation_count,
            "total_generation_time": self.total_generation_time,
            "average_generation_time": avg_generation_time,
            "max_tokens": self.max_tokens,
            "device": self.device,
            "offload_ratio": self.offload_ratio,
            "loaded": self.model is not None,
            "last_used": self.last_used,
            "recently_used": self.recently_used()
        }

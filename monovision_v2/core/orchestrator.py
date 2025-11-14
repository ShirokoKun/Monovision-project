"""
MonoVision V2 - Core Orchestrator
Implements the concrete tiered architecture with optimal resource allocation
Enhanced with proper pipeline architecture: Preprocessor -> Async Pipeline -> Fusion -> LLM -> Cache -> UI
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import torch
from PIL import Image
import io

from .resource_manager import ResourceManager
from .config_manager import ConfigManager
from ..vision.fusion_layer import VisionFusionLayer
from ..language.tier1_flan import FlanT5Processor
from ..language.tier2_phi import Phi2Processor
from ..language.tier3_api import ThirdPartyAPIProcessor
from ..memory.cache_manager import CacheManager
from ..nlp.context_prompts import ContextAwarePrompts
from ..preprocessing.image_preprocessor import ImagePreprocessor
from ..fusion.vision_fusion_module import VisionFusionModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Processing modes matching the concrete architecture"""
    FAST = "fast"           # BLIP + Flan-T5 (≤20 tokens)
    BALANCED = "balanced"   # BLIP + CLIP + Phi-2 (50 tokens)
    RICH = "rich"          # BLIP + CLIP + Mistral-7B (50-100 tokens)

@dataclass
class ProcessingRequest:
    """Request structure for the orchestrator"""
    image_data: bytes
    query: str
    mode: ProcessingMode
    session_id: str
    request_id: str
    include_objects: bool = False
    send_image_to_gemini: bool = False

@dataclass
class ProcessingResponse:
    """Response structure from the orchestrator"""
    request_id: str
    mode: ProcessingMode
    processing_time: float
    vision_results: Dict[str, Any]
    language_response: str
    token_count: int
    cached: bool
    error: Optional[str] = None

class MonoVisionOrchestrator:
    """
    Main orchestrator implementing the concrete tiered architecture
    Manages GPU/CPU allocation and routes requests to appropriate models
    """
    
    def __init__(self):
        self.config = ConfigManager()
        self.resource_manager = ResourceManager()
        self.cache_manager = CacheManager()
        
        # Vision components (GPU resident)
        self.vision_fusion = None
        
        # Language components (tiered loading)
        self.flan_processor = None
        self.phi2_processor = None
        self.api_processor = None
        
        # V3 Enhancement: Context-aware prompts
        self.context_prompts = ContextAwarePrompts()
        
        # NEW: Pipeline Architecture Components
        self.image_preprocessor = ImagePreprocessor()
        self.fusion_module = VisionFusionModule()
        
        # Processing queue for GPU coordination
        self.processing_queue = asyncio.Queue()
        self.is_processing = False
        
        # Performance tracking
        self.request_count = 0
        self.total_processing_time = 0.0
        
        logger.info("MonoVision V2 Orchestrator initialized")
    
    async def initialize(self):
        """Initialize all components according to the concrete architecture"""
        logger.info("🚀 Initializing MonoVision V2 - Concrete Architecture")
        
        # 1. Initialize Cache Manager first
        logger.info("💾 Initializing Cache Manager")
        await self.cache_manager.initialize()
        
        # 2. Initialize GPU-resident vision models
        logger.info("📸 Loading vision models (GPU resident)")
        self.vision_fusion = VisionFusionLayer(resource_manager=self.resource_manager)
        # Pass cache_manager to vision_fusion
        self.vision_fusion.cache_manager = self.cache_manager
        await self.vision_fusion.initialize()
        
        # 3. Initialize Tier 1 (Fast) - Flan-T5
        logger.info("⚡ Loading Tier 1: Flan-T5-Small (GPU resident)")
        self.flan_processor = FlanT5Processor()
        await self.flan_processor.initialize()
        
        # 4. Reserve GPU memory allocation
        await self.resource_manager.allocate_gpu_memory()
        
        logger.info("✅ Core components initialized successfully")
        logger.info(f"🎯 GPU Memory Usage: {self.resource_manager.get_gpu_usage()}")
    
    def detect_processing_mode(self, query: str, user_preference: Optional[str] = None) -> ProcessingMode:
        """
        Intelligent mode detection based on query complexity and user preference
        Following the concrete architecture routing rules
        """
        if user_preference:
            mode_map = {
                "fast": ProcessingMode.FAST,
                "balanced": ProcessingMode.BALANCED,
                "rich": ProcessingMode.RICH,
                "detailed": ProcessingMode.RICH,
                "full": ProcessingMode.RICH
            }
            if user_preference.lower() in mode_map:
                return mode_map[user_preference.lower()]
        
        # Automatic detection based on query patterns
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Rich mode indicators
        rich_keywords = [
            "detailed", "analyze", "comprehensive", "explain", "context",
            "meaning", "interpretation", "full analysis", "breakdown",
            "describe thoroughly", "what might be", "implications"
        ]
        
        # Fast mode indicators
        fast_keywords = [
            "what is", "identify", "name", "quick", "simple", "basic", "brief"
        ]
        
        if any(keyword in query_lower for keyword in rich_keywords) or word_count > 8:
            return ProcessingMode.RICH
        elif any(keyword in query_lower for keyword in fast_keywords) or word_count <= 3:
            return ProcessingMode.FAST
        else:
            return ProcessingMode.BALANCED
    
    async def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """
        Main processing pipeline following the concrete architecture
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self.cache_manager.generate_cache_key(request.image_data, request.query, request.mode)
            cached_result = await self.cache_manager.get_cached_result(cache_key)
            
            if cached_result:
                logger.info(f"⚡ Cache hit for request {request.request_id}")
                # Handle cached result - it might be a dict or ProcessingResponse
                if isinstance(cached_result, dict):
                    # Convert dict to ProcessingResponse
                    response = ProcessingResponse(
                        request_id=request.request_id,
                        mode=request.mode,
                        processing_time=cached_result.get('processing_time', 0.0),
                        vision_results=cached_result.get('vision_results', {}),
                        language_response=cached_result.get('language_response', ''),
                        token_count=cached_result.get('token_count', 0),
                        cached=True,
                        error=cached_result.get('error')
                    )
                else:
                    # Already a ProcessingResponse object
                    response = cached_result
                    response.request_id = request.request_id
                    response.cached = True
                return response
            
            # Add to processing queue
            await self.processing_queue.put(request)
            
            # Process the request
            response = await self._process_request_internal(request, start_time)
            
            # Cache the result
            await self.cache_manager.cache_result(cache_key, response)
            
            # Store session memory for context
            if response.language_response and not response.error:
                image_hash = cache_key.split('_')[0]  # Extract image hash from cache key
                await self.cache_manager.store_session_memory(
                    request.session_id, 
                    image_hash, 
                    request.query, 
                    response.language_response
                )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing request {request.request_id}: {e}")
            return ProcessingResponse(
                request_id=request.request_id,
                mode=request.mode,
                processing_time=time.time() - start_time,
                vision_results={},
                language_response=f"Error processing request: {str(e)}",
                token_count=0,
                cached=False,
                error=str(e)
            )
    
    async def _process_request_internal(self, request: ProcessingRequest, start_time: float) -> ProcessingResponse:
        """
        Internal processing following the NEW PIPELINE ARCHITECTURE:
        [ Image Upload ] -> [ Image Preprocessor ] -> [ Async Pipeline ] -> [ Fusion Module ] -> [ LLM ] -> [ Cache Check ] -> [ UI Renderer ]
        """
        
        logger.info(f"🚀 Starting pipeline processing for {request.request_id} in {request.mode.value} mode")
        
        # STEP 1: IMAGE PREPROCESSOR
        # ┌─────────────────┐
        # │ Image Preprocessor│
        # │ - Resize / Normalize │
        # │ - Optional Filters  │
        # └─────────┬─────────┘
        vision_results = {}
        preprocessed_data = None
        
        if request.image_data and len(request.image_data) > 0:
            logger.info("� Step 1: Image Preprocessing")
            
            # Validate image data before processing
            if not self._is_valid_image_data(request.image_data):
                logger.warning("⚠️ Invalid image data detected, skipping image preprocessing")
                preprocessed_data = None
            else:
                # Convert bytes to PIL Image
                try:
                    image = Image.open(io.BytesIO(request.image_data))
                    
                    # Preprocess image with quality enhancement for better vision results
                    preprocessed_data = self.image_preprocessor.preprocess_image(
                        image=image,
                        enhance_quality=(request.mode != ProcessingMode.FAST),  # Enhance for balanced/rich modes
                        apply_filters=(request.mode == ProcessingMode.RICH)     # Filters only for rich mode
                    )
                    
                    logger.info(f"✅ Image preprocessed: {preprocessed_data['metadata']['original_size']} -> {preprocessed_data['metadata']['processed_size']}")
                    
                except Exception as e:
                    logger.error(f"❌ Image preprocessing failed: {e}")
                    preprocessed_data = None
        else:
            logger.info("💬 Text-only mode: Skipping image preprocessing")
            # For text-only mode, we'll skip vision processing and go straight to language
        
        # STEP 2: ASYNC PIPELINE (PARALLEL VISION PROCESSING)
        #      ┌──────────────┐
        #      │ Async Pipeline│
        #      └──────┬───────┘
        #             │
        #    ┌────────┴─────────┐
        #    │                  │
        #    ▼                  ▼
        # ┌───────────┐      ┌─────────────┐
        # │  BLIP V2  │      │  YOLOv8     │
        # │ (Caption  │      │ (Object List│
        # │  / Semantic)│    │ & Bounding Boxes) │
        # └─────┬─────┘      └─────┬───────┘
        
        blip_result = {}
        yolo_result = []
        clip_result = {}
        
        if preprocessed_data:
            logger.info("📸 Step 2: Async Pipeline - Parallel Vision Processing")
            
            vision_start = time.time()
            processed_image = preprocessed_data['image']
            
            # Run vision models in parallel
            vision_tasks = []
            
            # Always run BLIP for caption
            vision_tasks.append(self._run_blip_analysis(processed_image))
            
            # Run CLIP for balanced/rich modes
            if request.mode in [ProcessingMode.BALANCED, ProcessingMode.RICH]:
                vision_tasks.append(self._run_clip_analysis(processed_image))
            
            # Run YOLO if objects requested
            if request.include_objects:
                vision_tasks.append(self._run_yolo_analysis(processed_image))
            
            # Execute all vision tasks in parallel
            vision_task_results = await asyncio.gather(*vision_tasks, return_exceptions=True)
            
            # Parse results
            blip_result = vision_task_results[0] if len(vision_task_results) > 0 and not isinstance(vision_task_results[0], Exception) else {}
            
            if len(vision_task_results) > 1 and not isinstance(vision_task_results[1], Exception):
                if request.mode in [ProcessingMode.BALANCED, ProcessingMode.RICH]:
                    clip_result = vision_task_results[1]
                elif request.include_objects:
                    yolo_result = vision_task_results[1]
            
            if len(vision_task_results) > 2 and not isinstance(vision_task_results[2], Exception):
                yolo_result = vision_task_results[2]
            
            vision_time = time.time() - vision_start
            logger.info(f"✅ Async pipeline completed in {vision_time:.2f}s")
        
        # STEP 3: FUSION MODULE
        #   ┌─────────────────────┐
        #   │ Fusion Module        │
        #   │ - Combine BLIP caption│
        #   │   + YOLO object list │
        #   │   + CLIP keywords    │
        #   └─────────┬───────────┘
        
        fusion_result = {}
        if blip_result or yolo_result or clip_result:
            logger.info("� Step 3: Vision Fusion Module")
            
            fusion_start = time.time()
            
            # Determine fusion mode based on processing tier
            fusion_mode = "concise" if request.mode == ProcessingMode.FAST else \
                         "detailed" if request.mode == ProcessingMode.BALANCED else "technical"
            
            fusion_result = self.fusion_module.fuse_vision_results(
                blip_result=blip_result,
                yolo_result=yolo_result,
                clip_result=clip_result,
                fusion_mode=fusion_mode,
                user_query=request.query
            )
            
            fusion_time = time.time() - fusion_start
            logger.info(f"✅ Vision fusion completed: quality={fusion_result.get('fusion_quality', 0):.2f}, time={fusion_time:.2f}s")
            
            # Build comprehensive vision results
            vision_results = {
                'caption': blip_result.get('caption', ''),
                'clip_keywords': clip_result.get('keywords', []),
                'objects': yolo_result,
                'fusion_result': fusion_result,
                'preprocessing_metadata': preprocessed_data['metadata'] if preprocessed_data else {},
                'vision_metrics': {
                    'blip_time': blip_result.get('processing_time', 0),
                    'clip_time': clip_result.get('processing_time', 0),
                    'yolo_time': getattr(yolo_result, 'processing_time', 0) if hasattr(yolo_result, 'processing_time') else 0,
                    'fusion_time': fusion_time if 'fusion_time' in locals() else 0,
                    'total_vision_time': vision_time if 'vision_time' in locals() else 0
                }
            }
        
        # STEP 4: LLM PROCESSING (TIER-BASED)
        #     ┌───────────────┐
        #     │ Phi-2 / LLM   │
        #     │ - Refine output│
        #     │ - Apply mode   │
        #     │   template     │
        #     └───────┬───────┘
        
        logger.info(f"🤖 Step 4: Language Processing - Mode: {request.mode.value}")
        language_start = time.time()
        
        # Retrieve session memory for context
        session_memory = await self.cache_manager.get_session_memory(request.session_id, limit=3)
        
        # Handle text-only mode with empty vision results
        if not preprocessed_data:
            logger.info("💬 Text-only processing - using minimal vision context")
            vision_results = {
                'caption': '',
                'clip_keywords': [],
                'objects': [],
                'fusion_result': {},
                'preprocessing_metadata': {},
                'vision_metrics': {}
            }
        
        # Generate context-aware prompt with fusion results and session memory
        context_prompt = self.context_prompts.generate_context_prompt(
            request.mode.value, 
            request.query, 
            vision_results,
            enable_fusion=True if preprocessed_data else False,  # Only enable fusion if we have image
            max_tokens=50 if request.mode == ProcessingMode.FAST else 100,
            session_memory=session_memory
        )
        
        # Process with appropriate language model
        if request.mode == ProcessingMode.FAST:
            # Tier 1: BLIP + Flan-T5 (≤50 tokens)
            if not self.flan_processor:
                self.flan_processor = FlanT5Processor()
                await self.flan_processor.initialize()
            
            language_response, token_count = await self.flan_processor.generate_response(
                vision_results, request.query, max_tokens=50, context_prompt=context_prompt
            )
            
        elif request.mode == ProcessingMode.BALANCED:
            # Tier 2: BLIP + CLIP + Phi-2 (50 tokens) - THIS SHOULD NOW WORK PROPERLY
            if not self.phi2_processor:
                logger.info("🔄 Loading Phi-2 (4-bit) for balanced mode")
                self.phi2_processor = Phi2Processor()
                await self.phi2_processor.initialize()
            
            # Pass enhanced vision results with fusion
            language_response, token_count = await self.phi2_processor.generate_response(
                vision_results, request.query, max_tokens=50, context_prompt=context_prompt
            )
            
        else:  # ProcessingMode.RICH
            # Tier 3: BLIP + CLIP + Third-Party API (50-100 tokens)
            if not self.api_processor:
                logger.info("🔄 Loading Third-Party API Processor for rich mode")
                self.api_processor = ThirdPartyAPIProcessor()
                await self.api_processor.initialize()

            # Pass image data for Gemini API if requested
            if request.send_image_to_gemini and request.image_data and self._is_valid_image_data(request.image_data):
                # Add image data to vision results for API processor
                vision_results['image_data'] = request.image_data
                vision_results['image_format'] = 'jpeg'  # Default format
                logger.info(f"🎯 Sending image data to Gemini API: {len(request.image_data)} bytes")
            else:
                logger.info(f"📝 Using text-only mode for Gemini API (send_image: {request.send_image_to_gemini}, has_data: {bool(request.image_data)}, valid: {self._is_valid_image_data(request.image_data) if request.image_data else False})")

            language_response, token_count = await self.api_processor.generate_response(
                vision_results, request.query, max_tokens=100, context_prompt=context_prompt
            )

            # Check for fallback request from API processor
            if language_response.startswith("FALLBACK_REQUESTED:"):
                logger.warning("🔄 API processor requested fallback - switching to balanced mode")
                # Extract the actual error message
                error_msg = language_response.replace("FALLBACK_REQUESTED:", "").strip()

                # Try balanced mode as fallback
                if not self.phi2_processor:
                    logger.info("🔄 Loading Phi-2 for fallback processing")
                    try:
                        self.phi2_processor = Phi2Processor()
                        await self.phi2_processor.initialize()
                        language_response, token_count = await self.phi2_processor.generate_response(
                            vision_results, request.query, max_tokens=50, context_prompt=context_prompt
                        )
                        logger.info("✅ Fallback to balanced mode successful")
                    except Exception as phi2_error:
                        logger.error(f"❌ Phi-2 fallback failed: {phi2_error}")
                        # Try Flan-T5 as final fallback
                        if not self.flan_processor:
                            logger.info("🔄 Loading Flan-T5 for final fallback")
                            try:
                                self.flan_processor = FlanT5Processor()
                                await self.flan_processor.initialize()
                                language_response, token_count = await self.flan_processor.generate_response(
                                    vision_results, request.query, max_tokens=30, context_prompt=context_prompt
                                )
                                logger.info("✅ Final fallback to fast mode successful")
                            except Exception as flan_error:
                                logger.error(f"❌ Flan-T5 fallback failed: {flan_error}")
                                language_response = f"Unable to process request: {error_msg}. All processing tiers failed."
                                token_count = 0
                        else:
                            # Flan-T5 already loaded
                            try:
                                language_response, token_count = await self.flan_processor.generate_response(
                                    vision_results, request.query, max_tokens=30, context_prompt=context_prompt
                                )
                                logger.info("✅ Final fallback to fast mode successful")
                            except Exception as flan_error:
                                logger.error(f"❌ Flan-T5 fallback failed: {flan_error}")
                                language_response = f"Unable to process request: {error_msg}. All processing tiers failed."
                                token_count = 0
                else:
                    # Phi-2 already loaded
                    try:
                        language_response, token_count = await self.phi2_processor.generate_response(
                            vision_results, request.query, max_tokens=50, context_prompt=context_prompt
                        )
                        logger.info("✅ Fallback to balanced mode successful")
                    except Exception as phi2_error:
                        logger.error(f"❌ Phi-2 fallback failed: {phi2_error}")
                        # Try Flan-T5 as final fallback
                        if not self.flan_processor:
                            logger.info("🔄 Loading Flan-T5 for final fallback")
                            try:
                                self.flan_processor = FlanT5Processor()
                                await self.flan_processor.initialize()
                                language_response, token_count = await self.flan_processor.generate_response(
                                    vision_results, request.query, max_tokens=30, context_prompt=context_prompt
                                )
                                logger.info("✅ Final fallback to fast mode successful")
                            except Exception as flan_error:
                                logger.error(f"❌ Flan-T5 fallback failed: {flan_error}")
                                language_response = f"Unable to process request: {error_msg}. All processing tiers failed."
                                token_count = 0
                        else:
                            # Flan-T5 already loaded
                            try:
                                language_response, token_count = await self.flan_processor.generate_response(
                                    vision_results, request.query, max_tokens=30, context_prompt=context_prompt
                                )
                                logger.info("✅ Final fallback to fast mode successful")
                            except Exception as flan_error:
                                logger.error(f"❌ Flan-T5 fallback failed: {flan_error}")
                                language_response = f"Unable to process request: {error_msg}. All processing tiers failed."
                                token_count = 0
        
        language_time = time.time() - language_start
        total_time = time.time() - start_time
        
        logger.info(f"✅ Language processing completed in {language_time:.2f}s")
        logger.info(f"🎯 PIPELINE COMPLETE: {total_time:.2f}s total")
        
        # Update performance metrics
        self.request_count += 1
        self.total_processing_time += total_time
        
        return ProcessingResponse(
            request_id=request.request_id,
            mode=request.mode,
            processing_time=total_time,
            vision_results=vision_results,
            language_response=language_response,
            token_count=token_count,
            cached=False
        )
    
    def _is_valid_image_data(self, image_data: bytes) -> bool:
        """
        Validate if the image data is a valid image format
        """
        if not image_data or len(image_data) < 10:
            return False
        
        # Check for common image file signatures
        # JPEG: FF D8
        # PNG: 89 50 4E 47
        # GIF: 47 49 46
        # BMP: 42 4D
        # WebP: 52 49 46 46 (RIFF)
        
        if len(image_data) >= 2:
            # JPEG
            if image_data[0] == 0xFF and image_data[1] == 0xD8:
                return True
            # PNG
            if len(image_data) >= 8 and image_data[:8] == b'\x89PNG\r\n\x1a\n':
                return True
            # GIF
            if image_data[:3] == b'GIF':
                return True
            # BMP
            if image_data[:2] == b'BM':
                return True
            # WebP
            if len(image_data) >= 12 and image_data[:4] == b'RIFF' and image_data[8:12] == b'WEBP':
                return True
        
        return False
    
    async def _run_blip_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Run BLIP analysis as part of async pipeline"""
        try:
            start_time = time.time()
            result = await self.vision_fusion._generate_blip_caption(image)
            processing_time = time.time() - start_time
            
            return {
                'caption': result,
                'processing_time': processing_time,
                'model': 'BLIP-base'
            }
        except Exception as e:
            logger.error(f"❌ BLIP analysis failed: {e}")
            return {'caption': '', 'processing_time': 0, 'error': str(e)}
    
    async def _run_clip_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Run CLIP analysis as part of async pipeline"""
        try:
            start_time = time.time()
            result = await self.vision_fusion._generate_clip_analysis(image)
            processing_time = time.time() - start_time
            
            return {
                'keywords': result.get('keywords', []),
                'embedding': result.get('embedding', []),
                'confidence_scores': result.get('confidence_scores', {}),
                'processing_time': processing_time,
                'model': 'CLIP-ViT-B/32'
            }
        except Exception as e:
            logger.error(f"❌ CLIP analysis failed: {e}")
            return {'keywords': [], 'embedding': [], 'processing_time': 0, 'error': str(e)}
    
    async def _run_yolo_analysis(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Run YOLO analysis as part of async pipeline"""
        try:
            start_time = time.time()
            result = await self.vision_fusion._detect_objects(image)
            processing_time = time.time() - start_time
            
            # Add processing metadata to each object
            for obj in result:
                obj['processing_time'] = processing_time
                obj['model'] = 'YOLOv8n'
            
            return result
        except Exception as e:
            logger.error(f"❌ YOLO analysis failed: {e}")
            return []
    
    async def cleanup_unused_models(self):
        """Clean up unused models to free memory"""
        logger.info("🧹 Cleaning up unused models")
        
        # Keep vision models and Flan-T5 (core components)
        # Unload Phi-2 and Mistral if not recently used
        if self.phi2_processor and not self.phi2_processor.recently_used():
            await self.phi2_processor.unload()
            self.phi2_processor = None
            logger.info("🗑️ Unloaded Phi-2")
        
        if self.api_processor and not self.api_processor.recently_used():
            await self.api_processor.unload()
            self.api_processor = None
            logger.info("🗑️ Unloaded Third-Party API Processor")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and performance metrics"""
        avg_processing_time = (
            self.total_processing_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        # Get GPU memory info safely
        try:
            gpu_memory = self.resource_manager.get_gpu_usage()
        except Exception as e:
            logger.warning(f"⚠️ Could not get GPU usage: {e}")
            gpu_memory = {"error": str(e)}
        
        # Get cache stats safely
        try:
            cache_stats = self.cache_manager.get_stats()
        except Exception as e:
            logger.warning(f"⚠️ Could not get cache stats: {e}")
            cache_stats = {"error": str(e)}
        
        return {
            "status": "operational",
            "architecture": "MonoVision V2 - Concrete Tiered",
            "request_count": self.request_count,
            "average_processing_time": avg_processing_time,
            "gpu_memory": gpu_memory,
            "models_loaded": {  # Changed from "loaded_models" to "models_loaded" 
                "vision_fusion": self.vision_fusion is not None,
                "flan_t5": self.flan_processor is not None,
                "phi2": self.phi2_processor is not None,
                "api_processor": self.api_processor is not None
            },
            "cache_stats": cache_stats,
            "processing_modes": [mode.value for mode in ProcessingMode]
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Alias for get_system_status for compatibility"""
        return self.get_system_status()
    
    async def cleanup(self):
        """Cleanup resources and models"""
        try:
            logger.info("🧹 Cleaning up MonoVision V2 resources...")
            
            # Cleanup cache manager
            if self.cache_manager:
                await self.cache_manager.cleanup()
            
            # Clear model references to free GPU memory
            self.vision_fusion = None
            self.flan_processor = None
            self.phi2_processor = None
            self.api_processor = None
            
            # Force garbage collection
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("✅ Cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")

# Global orchestrator instance
orchestrator = MonoVisionOrchestrator()

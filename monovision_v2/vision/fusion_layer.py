"""
Vision Fusion Layer for MonoVision V2
Implements the concrete vision architecture: BLIP + CLIP + YOLOv8n
"""

import asyncio
import logging
import time
import hashlib
from typing import Dict, Any, Optional, List, Tuple
import torch
import numpy as np
from PIL import Image
import io

# Import vision models
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from transformers import CLIPProcessor, CLIPModel
    from ultralytics import YOLO
    import cv2
except ImportError as e:
    logging.error(f"Missing vision dependencies: {e}")

from ..core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)

class VisionFusionLayer:
    """
    Vision Fusion Layer implementing the concrete architecture:
    - BLIP-base: Image captioning (GPU, fp16, ~800MB)
    - CLIP ViT-B/32: Semantic embeddings (GPU, fp16, ~500MB)  
    - YOLOv8n: Object detection (CPU-only, on-demand)
    
    Total GPU usage: ~1.3GB (within budget)
    """
    
    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        self.resource_manager = resource_manager or ResourceManager()
        
        # Cache manager (will be set by orchestrator)
        self.cache_manager = None
        
        # Model configurations from concrete architecture
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_fp16 = True and self.device == "cuda"
        self.image_size = 384  # Optimal for both BLIP and CLIP
        
        # Model instances (GPU resident)
        self.blip_processor = None
        self.blip_model = None
        self.clip_processor = None
        self.clip_model = None
        self.yolo_model = None  # CPU-only, loaded on demand
        
        # Performance tracking
        self.processing_count = 0
        self.total_processing_time = 0.0
        
        logger.info("📸 Vision Fusion Layer initialized")
    
    async def initialize(self):
        """Initialize vision models according to concrete architecture"""
        logger.info("🚀 Initializing Vision Models - Concrete Architecture")
        
        try:
            # 1. Initialize BLIP (Image Captioning) - GPU Resident
            await self._load_blip()
            
            # 2. Initialize CLIP (Semantic Embeddings) - GPU Resident  
            await self._load_clip()
            
            # 3. YOLOv8n stays unloaded (CPU-only, on-demand)
            logger.info("📦 YOLOv8n: CPU-only, will load on demand")
            
            # 4. Register memory allocations
            self.resource_manager.register_model_allocation("BLIP", 800)  # MB
            self.resource_manager.register_model_allocation("CLIP", 500)  # MB
            
            logger.info("✅ Vision models initialized successfully")
            logger.info(f"📊 GPU allocation: ~1.3GB (BLIP: 800MB + CLIP: 500MB)")
            
        except Exception as e:
            logger.error(f"❌ Error initializing vision models: {e}")
            raise
    
    async def _load_blip(self):
        """Load BLIP model for image captioning"""
        logger.info("📸 Loading BLIP-base (Image Captioning)")
        
        model_name = "Salesforce/blip-image-captioning-base"
        
        # Load processor
        self.blip_processor = BlipProcessor.from_pretrained(
            model_name,
            cache_dir="cache/hf"
        )
        
        # Load model with fp16 optimization
        model_kwargs = {
            "cache_dir": "cache/hf",
            "torch_dtype": torch.float16 if self.use_fp16 else torch.float32,
            "low_cpu_mem_usage": True
        }
        
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs
        ).to(self.device)
        
        # Set to eval mode
        self.blip_model.eval()
        
        logger.info(f"✅ BLIP loaded successfully on {self.device}")
    
    async def _load_clip(self):
        """Load CLIP model for semantic embeddings"""
        logger.info("🔗 Loading CLIP ViT-B/32 (Semantic Embeddings)")
        
        model_name = "openai/clip-vit-base-patch32"
        
        # Load processor
        self.clip_processor = CLIPProcessor.from_pretrained(
            model_name,
            cache_dir="cache/hf"
        )
        
        # Load model with fp16 optimization
        self.clip_model = CLIPModel.from_pretrained(
            model_name,
            cache_dir="cache/hf",
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32
        ).to(self.device)
        
        # Set to eval mode
        self.clip_model.eval()
        
        logger.info(f"✅ CLIP loaded successfully on {self.device}")
    
    async def _load_yolo_on_demand(self):
        """Load YOLOv8n on GPU when available, CPU as fallback"""
        if self.yolo_model is None:
            logger.info("📦 Loading YOLOv8n (Object Detection)")
            try:
                # Handle PyTorch 2.6 weights_only issue
                import torch
                original_weights_only = getattr(torch.serialization, '_weights_only_unpickler', None)
                
                # Temporarily allow unsafe loading for YOLOv8
                with torch.serialization.safe_globals(['ultralytics.nn.tasks.DetectionModel']):
                    self.yolo_model = YOLO("yolov8n.pt")
                
                # Try GPU first, fallback to CPU
                try:
                    if torch.cuda.is_available():
                        self.yolo_model.to("cuda")
                        logger.info("✅ YOLOv8n loaded on GPU")
                    else:
                        self.yolo_model.to("cpu")
                        logger.info("✅ YOLOv8n loaded on CPU (no GPU available)")
                except Exception as gpu_error:
                    logger.warning(f"⚠️ GPU loading failed, using CPU: {gpu_error}")
                    self.yolo_model.to("cpu")
                    logger.info("✅ YOLOv8n loaded on CPU (fallback)")
                    
            except Exception as e:
                # Fallback: try with weights_only=False
                try:
                    import torch
                    # Patch torch.load temporarily
                    original_load = torch.load
                    def patched_load(*args, **kwargs):
                        kwargs['weights_only'] = False
                        return original_load(*args, **kwargs)
                    torch.load = patched_load
                    
                    self.yolo_model = YOLO("yolov8n.pt")
                    
                    # Try GPU first, fallback to CPU
                    try:
                        if torch.cuda.is_available():
                            self.yolo_model.to("cuda")
                            logger.info("✅ YOLOv8n loaded on GPU (fallback method)")
                        else:
                            self.yolo_model.to("cpu")
                            logger.info("✅ YOLOv8n loaded on CPU (fallback method, no GPU)")
                    except Exception as gpu_error:
                        logger.warning(f"⚠️ GPU loading failed in fallback, using CPU: {gpu_error}")
                        self.yolo_model.to("cpu")
                        logger.info("✅ YOLOv8n loaded on CPU (fallback method)")
                    
                    # Restore original torch.load
                    torch.load = original_load
                    
                except Exception as e2:
                    logger.warning(f"⚠️ YOLOv8n loading failed: {e} / {e2}")
                    self.yolo_model = None
    
    async def process_image(self, image_data: bytes, include_objects: bool = False, mode: str = "balanced") -> Dict[str, Any]:
        """
        Enhanced processing pipeline with async optimization and fusion
        """
        start_time = time.time()
        
        try:
            # Convert mode to string if it's an enum
            if hasattr(mode, 'value'):
                mode = mode.value
            
            # Initialize variables for proper scope
            cached_clip = None
            clip_results = {"keywords": [], "embedding": []}
            clip_time = 0.0
            
            # 1. Preprocess image (CPU)
            image = await self._preprocess_image(image_data)
            
            # 2. Generate image hash for caching and similarity
            image_hash = hashlib.md5(image_data).hexdigest()[:16]
            
            # 3. Check for similar images in cache using CLIP embeddings
            similar_match = None
            if self.cache_manager:
                # Quick CLIP embedding for similarity check
                try:
                    quick_embedding = await self._generate_quick_clip_embedding(image)
                    similar_match = await self.cache_manager.find_similar_embeddings(
                        quick_embedding, similarity_threshold=0.85
                    )
                    if similar_match:
                        logger.info(f"🎯 Found similar cached image with {similar_match['similarity']:.3f} similarity")
                        # Use cached results but still run BLIP for this specific image
                except Exception as e:
                    logger.warning(f"⚠️ Similarity check failed: {e}")
            
            # 4. Parallel processing setup for optimal performance
            async def blip_task():
                """BLIP caption generation task"""
                caption_start = time.time()
                caption = await self._generate_blip_caption(image)
                caption_time = time.time() - caption_start
                return caption, caption_time
            
            async def yolo_task():
                """YOLO object detection task (if needed)"""
                if include_objects:
                    yolo_start = time.time()
                    objects = await self._detect_objects(image)
                    yolo_time = time.time() - yolo_start
                    return objects, yolo_time
                else:
                    return [], 0
            
            # 5. Execute BLIP and YOLO in parallel for better performance
            logger.info(f"🚀 Starting parallel vision processing (BLIP + YOLO)...")
            blip_result, yolo_result = await asyncio.gather(blip_task(), yolo_task())
            
            caption, caption_time = blip_result
            objects, yolo_time = yolo_result
            
            # 6. CLIP Semantic Analysis (GPU) - Enhanced with caching
            clip_results = {}
            clip_time = 0
            logger.info(f"🔍 Processing mode: {mode}, checking CLIP requirement...")
            if mode in ["balanced", "rich"]:
                logger.info(f"📊 CLIP analysis starting for mode: {mode}")
                clip_start = time.time()
                
                # Check if we have cached CLIP results
                cached_clip = None
                if self.cache_manager:
                    cached_clip = await self.cache_manager.get_cached_embedding(image_hash)
                
                if cached_clip:
                    clip_results = {
                        "keywords": cached_clip["keywords"],
                        "embedding": cached_clip["embedding"]
                    }
                    clip_time = 0.01  # Minimal time for cache hit
                    logger.info(f"✅ CLIP cache hit: {len(clip_results['keywords'])} keywords")
                else:
                    clip_results = await self._generate_clip_analysis(image)
                    clip_time = time.time() - clip_start
                    
                    # Cache the CLIP results
                    if self.cache_manager and "keywords" in clip_results and "embedding" in clip_results:
                        await self.cache_manager.cache_embedding(
                            image_hash, 
                            clip_results["embedding"], 
                            clip_results["keywords"]
                        )
                    
                    logger.info(f"📊 CLIP analysis completed: {len(clip_results.get('keywords', []))} keywords in {clip_time:.2f}s")
            else:
                logger.info(f"⏭️ Skipping CLIP for mode: {mode}")
            
            # 7. Enhanced Vision Fusion (V3 Feature)
            fusion_result = await self._perform_vision_fusion(
                caption=caption,
                clip_keywords=clip_results.get("keywords", []),
                objects=objects,
                mode=mode
            )
            
            # 8. Cache comprehensive results
            if self.cache_manager:
                await self.cache_manager.cache_vision_result(
                    image_hash=image_hash,
                    caption=caption,
                    keywords=clip_results.get("keywords", []),
                    objects=[obj.get("class", "unknown") for obj in objects]
                )
            
            total_time = time.time() - start_time
            
            # 9. Compile comprehensive results
            result = {
                "success": True,
                "image_hash": image_hash,
                "caption": caption,
                "keywords": clip_results.get("keywords", []),
                "objects": objects,
                "fusion_result": fusion_result,
                "embedding": clip_results.get("embedding", []),
                "similar_match": similar_match,
                "vision_metrics": {
                    "blip_time": caption_time,
                    "clip_time": clip_time,
                    "yolo_time": yolo_time,
                    "total_time": total_time,
                    "parallel_optimization": True
                },
                "processing_metadata": {
                    "mode": mode,
                    "include_objects": include_objects,
                    "cache_hit": similar_match is not None,
                    "clip_cached": cached_clip is not None,
                    "objects_detected": len(objects),
                    "keywords_extracted": len(clip_results.get("keywords", []))
                }
            }
            
            logger.info(f"✅ Vision processing complete: {total_time:.2f}s (BLIP: {caption_time:.2f}s, CLIP: {clip_time:.2f}s, YOLO: {yolo_time:.2f}s)")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in vision processing: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
            
            # 6. Fusion Layer - Create compact JSON summary
            fusion_results = self._create_fusion_summary(
                caption, clip_results, objects, image_hash
            )
            
            total_time = time.time() - start_time
            
            # 7. Performance tracking
            self.processing_count += 1
            self.total_processing_time += total_time
            
            # 8. Log performance
            logger.info(f"📸 Vision processing completed in {total_time:.2f}s")
            logger.info(f"    BLIP: {caption_time:.2f}s, CLIP: {clip_time:.2f}s, YOLO: {yolo_time:.2f}s")
            
            return fusion_results
            
        except Exception as e:
            logger.error(f"❌ Vision processing error: {e}")
            return {
                "error": str(e),
                "caption": "Error processing image",
                "objects": [],
                "clip_keywords": [],
                "processing_time": time.time() - start_time
            }
    
    async def _preprocess_image(self, image_data: bytes) -> Image.Image:
        """Preprocess image according to concrete architecture (CPU)"""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Resize to optimal size (384px as per architecture)
            image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"❌ Image preprocessing error: {e}")
            raise
    
    async def _generate_blip_caption(self, image: Image.Image) -> str:
        """Generate caption using BLIP (GPU)"""
        try:
            # Process image
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            # Generate caption with optimized parameters
            with torch.no_grad():
                if self.use_fp16:
                    with torch.autocast(device_type="cuda"):
                        outputs = self.blip_model.generate(
                            **inputs,
                            max_length=50,
                            num_beams=3,  # Balanced quality vs speed
                            do_sample=False,
                            early_stopping=True
                        )
                else:
                    outputs = self.blip_model.generate(
                        **inputs,
                        max_length=50,
                        num_beams=3,
                        do_sample=False,
                        early_stopping=True
                    )
            
            # Decode caption
            caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up caption
            caption = caption.strip()
            if caption.lower().startswith("a picture of"):
                caption = caption[12:].strip()
            if caption.lower().startswith("an image of"):
                caption = caption[11:].strip()
            
            return caption
            
        except Exception as e:
            logger.error(f"❌ BLIP caption generation error: {e}")
            return "Image caption unavailable"
    
    async def _generate_clip_analysis(self, image: Image.Image) -> Dict[str, Any]:
        """Generate enhanced semantic analysis using CLIP with expanded vocabulary"""
        try:
            logger.info("🔗 Starting CLIP semantic analysis...")
            
            # Expanded semantic keywords for deeper analysis
            keywords = [
                # Environmental & Setting
                "indoor", "outdoor", "urban", "rural", "natural", "artificial",
                "bright", "dark", "dimly lit", "well lit", "sunlit", "shadowy",
                
                # Mood & Atmosphere
                "peaceful", "busy", "calm", "energetic", "serene", "chaotic",
                "warm", "cool", "inviting", "intimidating", "cozy", "spacious",
                
                # Style & Aesthetics
                "modern", "vintage", "contemporary", "antique", "minimalist", "ornate",
                "colorful", "monochrome", "vibrant", "muted", "saturated", "pastel",
                
                # Composition & Perspective
                "close-up", "wide shot", "landscape", "portrait", "macro", "aerial",
                "static", "dynamic", "action", "still life", "candid", "posed",
                
                # Technical & Quality
                "sharp", "blurry", "detailed", "abstract", "realistic", "artistic",
                "professional", "amateur", "high quality", "documentary style",
                
                # Contextual
                "social", "solitary", "group", "individual", "formal", "casual",
                "commercial", "personal", "public", "private", "recreational", "work"
            ]
            
            logger.info(f"🔗 Processing {len(keywords)} enhanced keywords with CLIP...")
            
            # Process image and text
            inputs = self.clip_processor(
                text=keywords,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            logger.info("🔗 Running CLIP inference...")
            
            # Generate embeddings
            with torch.no_grad():
                if self.use_fp16:
                    with torch.autocast(device_type="cuda"):
                        outputs = self.clip_model(**inputs)
                else:
                    outputs = self.clip_model(**inputs)
                
                # Calculate similarities
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
            # Enhanced keyword selection with confidence thresholds
            confidence_threshold = 0.08  # Lowered threshold for better keyword detection
            high_confidence_threshold = 0.15  # For top-tier keywords
            
            # Get keywords with different confidence levels
            high_conf_indices = [i for i, prob in enumerate(probs) if prob > high_confidence_threshold]
            medium_conf_indices = [i for i, prob in enumerate(probs) if confidence_threshold < prob <= high_confidence_threshold]
            
            # Sort by confidence
            high_conf_keywords = [(keywords[i], probs[i]) for i in high_conf_indices]
            medium_conf_keywords = [(keywords[i], probs[i]) for i in medium_conf_indices]
            
            high_conf_keywords.sort(key=lambda x: x[1], reverse=True)
            medium_conf_keywords.sort(key=lambda x: x[1], reverse=True)
            
            # Select diverse keywords (max 8 for enhanced analysis)
            top_keywords = []
            confidence_scores = {}
            
            # Add high confidence keywords first (up to 5)
            for keyword, confidence in high_conf_keywords[:5]:
                top_keywords.append(keyword)
                confidence_scores[keyword] = float(confidence)
            
            # Add medium confidence keywords if we have space (up to 3 more)
            for keyword, confidence in medium_conf_keywords[:3]:
                if len(top_keywords) < 8:
                    top_keywords.append(keyword)
                    confidence_scores[keyword] = float(confidence)
            
            logger.info(f"🔗 CLIP found {len(top_keywords)} meaningful keywords: {top_keywords}")
            
            # Get image embedding for future use
            image_embedding = outputs.image_embeds.cpu().numpy()[0]
            
            # Enhanced metadata
            semantic_categories = self._categorize_keywords(top_keywords)
            
            return {
                "keywords": top_keywords,
                "embedding": image_embedding.tolist(),
                "confidence_scores": confidence_scores,
                "semantic_categories": semantic_categories,
                "analysis_depth": "enhanced" if len(top_keywords) >= 5 else "standard"
            }
            
        except Exception as e:
            logger.error(f"❌ CLIP analysis error: {e}")
            return {"keywords": [], "embedding": [], "confidence_scores": {}, "semantic_categories": {}}
    
    def _categorize_keywords(self, keywords: List[str]) -> Dict[str, List[str]]:
        """Categorize keywords for better semantic understanding"""
        categories = {
            "environment": [],
            "mood": [],
            "style": [],
            "composition": [],
            "technical": [],
            "context": []
        }
        
        category_mappings = {
            "environment": ["indoor", "outdoor", "urban", "rural", "natural", "artificial", "bright", "dark", "sunlit", "shadowy"],
            "mood": ["peaceful", "busy", "calm", "energetic", "serene", "chaotic", "warm", "cool", "inviting", "cozy"],
            "style": ["modern", "vintage", "contemporary", "antique", "minimalist", "ornate", "colorful", "monochrome", "vibrant"],
            "composition": ["close-up", "wide shot", "landscape", "portrait", "macro", "aerial", "static", "dynamic", "action"],
            "technical": ["sharp", "blurry", "detailed", "abstract", "realistic", "artistic", "professional", "high quality"],
            "context": ["social", "solitary", "group", "individual", "formal", "casual", "commercial", "personal", "recreational"]
        }
        
        for keyword in keywords:
            for category, category_keywords in category_mappings.items():
                if keyword in category_keywords:
                    categories[category].append(keyword)
                    break
        
        return {cat: kws for cat, kws in categories.items() if kws}
    
    async def _detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects using YOLOv8n with ENHANCED filtering and categorization"""
        try:
            # Load YOLO on demand
            await self._load_yolo_on_demand()

            if self.yolo_model is None:
                return []

            # Convert PIL to OpenCV format
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Run detection
            results = self.yolo_model(image_cv, verbose=False)

            # Parse results
            objects = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]

                        # Enhanced detection with lower threshold for better object coverage
                        if confidence > 0.25:  # Lowered from 0.5 to 0.25 for better detection
                            # Calculate object size for depth analysis
                            bbox_area = (x2 - x1) * (y2 - y1)
                            image_area = image.size[0] * image.size[1]
                            size_ratio = bbox_area / image_area

                            # Enhanced object information
                            objects.append({
                                "name": class_name,
                                "confidence": confidence,
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "size_ratio": float(size_ratio),
                                "center_point": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                                "area": float(bbox_area)
                            })

            # Apply ENHANCED OBJECT DETECTION PROCESSING
            enhanced_objects = self._apply_enhanced_object_detection(objects, image.size)

            return enhanced_objects

        except Exception as e:
            logger.error(f"❌ Object detection error: {e}")
            return []
    
    def _apply_enhanced_object_detection(self, objects: List[Dict[str, Any]], image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Apply ENHANCED object detection with improved accuracy and filtering
        This runs BEFORE Gemini API call for better context
        """
        if not objects:
            return []

        logger.info(f"🔍 Applying enhanced object detection to {len(objects)} raw detections")

        filtered_objects = []
        image_area = image_size[0] * image_size[1]

        # Step 1: Enhanced filtering with confidence and size validation
        for obj in objects:
            confidence = obj.get('confidence', 0)
            size_ratio = obj.get('size_ratio', 0)

            # Enhanced confidence threshold (35% minimum)
            if confidence < 0.35:
                continue

            # Size validation (minimum 1% of image, maximum 90%)
            if size_ratio < 0.01 or size_ratio > 0.9:
                continue

            # Add enhanced metadata
            obj['size_category'] = self._categorize_object_size(size_ratio)
            obj['position'] = self._get_object_position(obj.get('bbox', []), image_size)
            obj['description'] = self._get_object_description(obj.get('name', ''), confidence)

            filtered_objects.append(obj)

        logger.info(f"✅ After filtering: {len(filtered_objects)} objects (from {len(objects)})")

        # Step 2: Remove duplicates using bounding box overlap detection
        filtered_objects = self._remove_duplicate_objects(filtered_objects)

        # Step 3: Sort by confidence and limit objects
        filtered_objects = sorted(filtered_objects, key=lambda x: x.get('confidence', 0), reverse=True)
        filtered_objects = filtered_objects[:15]  # Maximum 15 objects

        # Step 4: Add categorization for better Gemini context
        categorized_objects = self._categorize_objects_for_gemini(filtered_objects)

        logger.info(f"🎯 Enhanced object detection complete: {len(categorized_objects)} objects with rich metadata")

        return categorized_objects

    def _categorize_object_size(self, size_ratio: float) -> str:
        """Categorize object size based on relative area"""
        if size_ratio > 0.3:
            return "dominant"
        elif size_ratio > 0.1:
            return "prominent"
        elif size_ratio > 0.02:
            return "moderate"
        else:
            return "small"

    def _get_object_position(self, bbox: List[float], image_size: Tuple[int, int]) -> str:
        """Determine object position in the image"""
        if not bbox or len(bbox) < 4:
            return "unknown"

        img_width, img_height = image_size
        x1, y1, x2, y2 = bbox

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Determine horizontal position
        if center_x < img_width * 0.33:
            h_pos = "left"
        elif center_x > img_width * 0.67:
            h_pos = "right"
        else:
            h_pos = "center"

        # Determine vertical position
        if center_y < img_height * 0.33:
            v_pos = "top"
        elif center_y > img_height * 0.67:
            v_pos = "bottom"
        else:
            v_pos = "middle"

        if h_pos == "center" and v_pos == "middle":
            return "center"
        else:
            return f"{v_pos}-{h_pos}"

    def _get_object_description(self, class_name: str, confidence: float) -> str:
        """Generate descriptive text for detected object"""
        confidence_text = "high" if confidence > 0.7 else "medium" if confidence > 0.5 else "low"
        return f"{class_name} (confidence: {confidence_text})"

    def _remove_duplicate_objects(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate objects based on class and position similarity"""
        if not objects:
            return []

        unique_objects = []
        for obj in objects:
            is_duplicate = False
            for existing_obj in unique_objects:
                # Check if objects are the same class and overlap significantly
                if (obj.get('name') == existing_obj.get('name') and
                    self._calculate_iou(obj.get('bbox', []), existing_obj.get('bbox', [])) > 0.5):
                    # Keep the one with higher confidence
                    if obj.get('confidence', 0) > existing_obj.get('confidence', 0):
                        unique_objects.remove(existing_obj)
                        break
                    else:
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_objects.append(obj)

        return unique_objects

    def _categorize_objects_for_gemini(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Categorize objects for better Gemini API context"""
        if not objects:
            return []

        # Group objects by category for richer analysis
        categories = {
            'people': [],
            'vehicles': [],
            'animals': [],
            'food': [],
            'furniture': [],
            'electronics': [],
            'other': []
        }

        # Categorization mappings
        category_mappings = {
            'people': ['person', 'man', 'woman', 'child', 'boy', 'girl', 'baby'],
            'vehicles': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'train', 'boat', 'airplane'],
            'animals': ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'],
            'food': ['apple', 'banana', 'orange', 'broccoli', 'carrot', 'pizza', 'cake', 'sandwich', 'hot dog'],
            'furniture': ['chair', 'table', 'sofa', 'bed', 'tv', 'monitor', 'keyboard', 'mouse'],
            'electronics': ['cell phone', 'laptop', 'remote', 'keyboard', 'mouse', 'microwave', 'oven', 'toaster']
        }

        for obj in objects:
            obj_name = obj.get('name', '').lower()
            categorized = False

            for category, keywords in category_mappings.items():
                if any(keyword in obj_name for keyword in keywords):
                    categories[category].append(obj)
                    categorized = True
                    break

            if not categorized:
                categories['other'].append(obj)

        # Add category information to objects
        categorized_objects = []
        for category, objs in categories.items():
            for obj in objs:
                obj['category'] = category
                obj['category_description'] = self._get_category_description(category, len(objs))
                categorized_objects.append(obj)

        return categorized_objects

    def _get_category_description(self, category: str, count: int) -> str:
        """Get descriptive text for object categories"""
        descriptions = {
            'people': f"person ({count} detected)" if count == 1 else f"people ({count} detected)",
            'vehicles': f"vehicle ({count} detected)" if count == 1 else f"vehicles ({count} detected)",
            'animals': f"animal ({count} detected)" if count == 1 else f"animals ({count} detected)",
            'food': f"food item ({count} detected)" if count == 1 else f"food items ({count} detected)",
            'furniture': f"furniture ({count} detected)" if count == 1 else f"furniture items ({count} detected)",
            'electronics': f"electronic device ({count} detected)" if count == 1 else f"electronic devices ({count} detected)",
            'other': f"object ({count} detected)" if count == 1 else f"objects ({count} detected)"
        }
        return descriptions.get(category, f"{category} ({count} detected)")
    
    async def _perform_vision_fusion(self, caption: str, clip_keywords: List[str], objects: List[Dict], mode: str) -> Dict[str, Any]:
        """
        Enhanced vision fusion combining BLIP, CLIP, and YOLO results
        """
        try:
            # Extract object class names
            object_names = [obj.get("name", obj.get("class", "unknown")) for obj in objects if obj.get("confidence", 0) > 0.3]
            
            # Fusion logic based on mode
            if mode == "rich" and clip_keywords and object_names:
                # Rich fusion: Combine all sources intelligently
                fused_description = self._create_rich_fusion(caption, clip_keywords, object_names)
            elif mode == "balanced" and (clip_keywords or object_names):
                # Balanced fusion: Enhance caption with available data
                fused_description = self._create_balanced_fusion(caption, clip_keywords, object_names)
            else:
                # Fast mode: Use caption as-is
                fused_description = caption
            
            # Quality scoring
            fusion_quality = self._calculate_fusion_quality(caption, clip_keywords, objects)
            
            return {
                "original_caption": caption,
                "enhanced_description": fused_description,
                "fusion_quality": fusion_quality,
                "fusion_metadata": {
                    "mode": mode,
                    "sources_used": self._get_sources_used(clip_keywords, object_names),
                    "enhancement_score": len(clip_keywords) + len(object_names)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error in vision fusion: {e}")
            return {
                "original_caption": caption,
                "enhanced_description": caption,
                "fusion_quality": 0.5,
                "error": str(e)
            }
    
    def _create_rich_fusion(self, caption: str, keywords: List[str], objects: List[str]) -> str:
        """Create rich fusion combining all vision sources with enhanced depth analysis"""
        # Start with base caption
        enhanced = caption
        
        # Enhanced object context with spatial awareness
        if objects:
            unique_objects = list(set(objects))
            object_count = len(unique_objects)
            
            if object_count == 1:
                enhanced += f" The image prominently features a {unique_objects[0]}, making it the central focus of the composition."
            elif object_count <= 3:
                enhanced += f" The scene contains several key elements: {', '.join(unique_objects[:-1])} and {unique_objects[-1]}, creating a complex visual narrative."
            else:
                primary = unique_objects[:3]
                enhanced += f" This is a detailed scene with multiple objects including {', '.join(primary)} among {object_count} total detected elements, suggesting a rich, layered composition."
        
        # Enhanced semantic context from CLIP with depth insights
        if keywords:
            # Categorize keywords for better understanding
            mood_keywords = [kw for kw in keywords if kw in ['bright', 'dark', 'peaceful', 'busy', 'colorful', 'monochrome']]
            style_keywords = [kw for kw in keywords if kw in ['modern', 'vintage', 'natural', 'artificial']]
            setting_keywords = [kw for kw in keywords if kw in ['indoor', 'outdoor', 'landscape', 'portrait']]
            action_keywords = [kw for kw in keywords if kw in ['static', 'dynamic', 'action', 'close-up']]
            
            # Add contextual depth based on keyword categories
            if mood_keywords:
                enhanced += f" The overall mood appears {', '.join(mood_keywords)}, contributing to the image's emotional impact."
            
            if style_keywords:
                enhanced += f" The aesthetic style suggests {', '.join(style_keywords)} characteristics."
            
            if setting_keywords:
                enhanced += f" This is captured in an {', '.join(setting_keywords)} setting."
            
            if action_keywords:
                enhanced += f" The composition style is {', '.join(action_keywords)}, affecting how viewers engage with the image."
        
        return enhanced
    
    def _create_balanced_fusion(self, caption: str, keywords: List[str], objects: List[str]) -> str:
        """Create balanced fusion with moderate enhancement and spatial context"""
        enhanced = caption
        
        # Enhanced object context with better depth analysis
        if objects:
            unique_objects = list(set(objects))
            object_count = len(unique_objects)
            
            if object_count == 1:
                enhanced += f" The main focus is a {unique_objects[0]}, which dominates the scene."
            elif object_count <= 3:
                enhanced += f" Key elements include {', '.join(unique_objects)}, creating visual interest."
            else:
                primary = unique_objects[:2]
                enhanced += f" The scene features {', '.join(primary)} among {object_count} detected objects, indicating complexity."
        
        # Enhanced semantic descriptors with categorization
        if keywords:
            # Prioritize mood and setting keywords for balanced mode
            mood_keywords = [kw for kw in keywords if kw in ['bright', 'dark', 'peaceful', 'busy', 'colorful', 'monochrome']]
            setting_keywords = [kw for kw in keywords if kw in ['indoor', 'outdoor', 'modern', 'vintage', 'natural']]
            
            key_descriptors = mood_keywords[:1] + setting_keywords[:1]
            key_descriptors = [kw for kw in key_descriptors if kw.lower() not in enhanced.lower()]
            
            if key_descriptors:
                if len(key_descriptors) == 1:
                    enhanced += f" The scene appears {key_descriptors[0]}."
                else:
                    enhanced += f" The atmosphere is {key_descriptors[0]} with {key_descriptors[1]} characteristics."
        
        return enhanced
    
    def _calculate_fusion_quality(self, caption: str, keywords: List[str], objects: List[Dict]) -> float:
        """Calculate enhanced fusion quality score (0.0 to 1.0) with depth analysis"""
        score = 0.2  # Base score for having a caption
        
        # Enhanced caption quality assessment
        if caption:
            caption_words = len(caption.split())
            if caption_words >= 8:
                score += 0.2  # Detailed caption bonus
            elif caption_words >= 5:
                score += 0.1  # Moderate caption bonus
            
            # Bonus for descriptive words
            descriptive_words = ['showing', 'with', 'featuring', 'containing', 'displaying']
            if any(word in caption.lower() for word in descriptive_words):
                score += 0.05
        
        # Enhanced keyword scoring with categorization
        if keywords:
            keyword_bonus = min(len(keywords) * 0.08, 0.35)  # Up to 0.35 for keywords
            score += keyword_bonus
            
            # Bonus for keyword diversity (different categories)
            categories = self._categorize_keywords(keywords)
            category_diversity = len(categories)
            if category_diversity >= 3:
                score += 0.1  # High diversity bonus
            elif category_diversity >= 2:
                score += 0.05  # Moderate diversity bonus
        
        # Enhanced object scoring with spatial analysis
        if objects:
            confident_objects = [obj for obj in objects if obj.get("confidence", 0) > 0.4]
            high_conf_objects = [obj for obj in objects if obj.get("confidence", 0) > 0.7]
            
            # Base object score
            object_score = min(len(confident_objects) * 0.1, 0.3)
            score += object_score
            
            # High confidence bonus
            if high_conf_objects:
                score += min(len(high_conf_objects) * 0.05, 0.15)
            
            # Spatial diversity bonus (objects in different areas)
            if len(confident_objects) >= 2:
                has_spatial_diversity = self._check_spatial_diversity(confident_objects)
                if has_spatial_diversity:
                    score += 0.08
        
        # Synergy bonus: when all components work together
        if caption and keywords and objects:
            # Check if objects mentioned in caption
            object_names = [obj.get("name", "") for obj in objects]
            caption_lower = caption.lower()
            mentioned_objects = [name for name in object_names if name.lower() in caption_lower]
            
            if mentioned_objects:
                score += 0.1  # Synergy bonus
        
        return min(score, 1.0)
    
    def _check_spatial_diversity(self, objects: List[Dict]) -> bool:
        """Check if objects are spatially diverse (not clustered)"""
        try:
            if len(objects) < 2:
                return False
            
            # Get center points
            centers = []
            for obj in objects:
                if "center_point" in obj:
                    centers.append(obj["center_point"])
                elif "bbox" in obj and len(obj["bbox"]) >= 4:
                    x1, y1, x2, y2 = obj["bbox"][:4]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    centers.append([center_x, center_y])
            
            if len(centers) < 2:
                return False
            
            # Calculate average distance between centers
            total_distance = 0
            comparisons = 0
            
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    distance = (dx * dx + dy * dy) ** 0.5
                    total_distance += distance
                    comparisons += 1
            
            if comparisons > 0:
                avg_distance = total_distance / comparisons
                # If average distance is greater than 100 pixels, consider it diverse
                return avg_distance > 100
            
            return False
        except:
            return False
    
    def _get_sources_used(self, keywords: List[str], objects: List[str]) -> List[str]:
        """Get list of vision sources used in fusion"""
        sources = ["BLIP"]  # Always have BLIP
        if keywords:
            sources.append("CLIP")
        if objects:
            sources.append("YOLO")
        return sources
    
    async def _generate_quick_clip_embedding(self, image: Image.Image) -> List[float]:
        """Generate quick CLIP embedding for similarity check"""
        try:
            if not self.clip_model or not self.clip_processor:
                return []
            
            with torch.no_grad():
                # Process PIL image using CLIP processor
                inputs = self.clip_processor(images=image, return_tensors="pt")
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate embedding
                image_features = self.clip_model.get_image_features(**inputs)
                embedding = image_features.cpu().numpy().flatten().tolist()
                return embedding
                
        except Exception as e:
            logger.error(f"❌ Error generating quick CLIP embedding: {e}")
            return []
    
    async def cleanup(self):
        """Clean up vision models"""
        logger.info("🧹 Cleaning up vision models")
        
        # Unload models
        if self.blip_model:
            del self.blip_model
            self.blip_model = None
        
        if self.blip_processor:
            del self.blip_processor
            self.blip_processor = None
        
        if self.clip_model:
            del self.clip_model
            self.clip_model = None
        
        if self.clip_processor:
            del self.clip_processor
            self.clip_processor = None
        
        if self.yolo_model:
            del self.yolo_model
            self.yolo_model = None
        
        # Unregister memory allocations
        self.resource_manager.unregister_model_allocation("BLIP")
        self.resource_manager.unregister_model_allocation("CLIP")
        
        # Clear GPU cache
        await self.resource_manager.cleanup_gpu_memory()
        
        logger.info("✅ Vision models cleanup completed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vision processing statistics"""
        return {
            "processing_count": self.processing_count,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.total_processing_time / self.processing_count if self.processing_count > 0 else 0,
            "models_loaded": {
                "blip": self.blip_model is not None,
                "clip": self.clip_model is not None,
                "yolo": self.yolo_model is not None
            },
            "device": self.device,
            "fp16_enabled": self.use_fp16,
            "image_size": self.image_size
        }

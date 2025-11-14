"""
Enhanced Fusion Module for MonoVision V3
Implements the fusion architecture: BLIP + YOLO + CLIP -> Enhanced Description
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import time
import numpy as np

logger = logging.getLogger(__name__)

class VisionFusionModule:
    """
    Advanced fusion module that combines:
    - BLIP V2 captions (semantic understanding)
    - YOLOv8 object detection (precise object identification)
    - CLIP embeddings (semantic context)
    
    According to the pipeline architecture diagram
    """
    
    def __init__(self):
        self.fusion_templates = {
            "detailed": "In this image, I can see {blip_caption}. The scene contains {object_list}. {spatial_context} {quality_context}",
            "concise": "{blip_caption} with {object_count} detected objects including {main_objects}.",
            "technical": "Visual analysis: {blip_caption}. Object detection identified: {detailed_objects}. Scene characteristics: {scene_analysis}."
        }
        
        # Confidence thresholds
        self.object_confidence_threshold = 0.3
        self.clip_keyword_threshold = 0.15
        
        logger.info("🔀 Vision Fusion Module initialized")
    
    def fuse_vision_results(
        self,
        blip_result: Dict[str, Any],
        yolo_result: List[Dict[str, Any]],
        clip_result: Dict[str, Any],
        fusion_mode: str = "detailed",
        user_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main fusion function that combines all vision model outputs
        
        Args:
            blip_result: BLIP caption and metadata
            yolo_result: YOLO object detection results (now enhanced)
            clip_result: CLIP keywords and embeddings
            fusion_mode: Type of fusion ("detailed", "concise", "technical")
            user_query: User's question to guide fusion
            
        Returns:
            Enhanced fusion result with metadata
        """
        try:
            start_time = time.time()
            logger.info(f"🔀 Starting vision fusion in '{fusion_mode}' mode")
            
            # Extract components
            blip_caption = blip_result.get('caption', '').strip()
            objects = self._process_enhanced_objects(yolo_result)  # Updated to handle enhanced objects
            clip_keywords = clip_result.get('keywords', [])
            
            # Analyze spatial relationships with enhanced data
            spatial_analysis = self._analyze_spatial_relationships(objects)
            
            # Generate scene context from CLIP
            scene_context = self._generate_scene_context(clip_keywords, blip_caption)
            
            # Create enhanced description with rich object context
            enhanced_description = self._create_enhanced_description(
                blip_caption=blip_caption,
                objects=objects,
                spatial_analysis=spatial_analysis,
                scene_context=scene_context,
                clip_keywords=clip_keywords,
                fusion_mode=fusion_mode,
                user_query=user_query
            )
            
            # Calculate fusion quality with enhanced object scoring
            fusion_quality = self._calculate_fusion_quality(
                blip_result, objects, clip_keywords, enhanced_description
            )
            
            # Generate fusion metadata with enhanced object statistics
            fusion_metadata = {
                "fusion_mode": fusion_mode,
                "processing_time": time.time() - start_time,
                "component_count": {
                    "blip_words": len(blip_caption.split()) if blip_caption else 0,
                    "detected_objects": len(objects),
                    "clip_keywords": len(clip_keywords),
                    "high_confidence_objects": len([obj for obj in objects if obj.get('confidence', 0) > 0.7]),
                    "categorized_objects": len([obj for obj in objects if obj.get('category')]),
                    "dominant_objects": len([obj for obj in objects if obj.get('size_category') == 'dominant'])
                },
                "spatial_relationships": len(spatial_analysis),
                "scene_characteristics": scene_context.get('characteristics', []),
                "fusion_confidence": fusion_quality,
                "object_categories": self._get_object_category_summary(objects)
            }
            
            result = {
                "enhanced_description": enhanced_description,
                "fusion_quality": fusion_quality,
                "fusion_metadata": fusion_metadata,
                "components": {
                    "original_caption": blip_caption,
                    "processed_objects": objects,
                    "spatial_analysis": spatial_analysis,
                    "scene_context": scene_context,
                    "clip_keywords": clip_keywords[:8]  # Top 8 keywords
                },
                "recommendations": self._generate_recommendations(fusion_quality, fusion_metadata)
            }
            
            logger.info(f"✅ Fusion complete: quality={fusion_quality:.2f}, objects={len(objects)}, categories={fusion_metadata['component_count']['categorized_objects']}, time={fusion_metadata['processing_time']:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Vision fusion failed: {e}")
            # Return fallback result
            return {
                "enhanced_description": blip_result.get('caption', 'Unable to analyze image'),
                "fusion_quality": 0.0,
                "fusion_metadata": {"error": str(e)},
                "components": {},
                "recommendations": ["Fusion failed, using BLIP caption only"]
            }
    
    def _process_objects(self, yolo_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and filter YOLO objects with enhanced metadata
        """
        processed_objects = []
        
        for obj in yolo_objects:
            confidence = obj.get('confidence', 0)
            if confidence >= self.object_confidence_threshold:
                processed_obj = {
                    "name": obj.get('name', obj.get('class', 'unknown')),
                    "confidence": confidence,
                    "bbox": obj.get('bbox', []),
                    "area": self._calculate_object_area(obj.get('bbox', [])),
                    "position": self._categorize_position(obj.get('bbox', [])),
                    "size_category": obj.get('size_category', 'unknown')
                }
                processed_objects.append(processed_obj)
        
        # Sort by confidence
        processed_objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"🔀 Processed {len(processed_objects)}/{len(yolo_objects)} objects above confidence threshold")
        return processed_objects
    
    def _process_enhanced_objects(self, yolo_objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process enhanced objects from vision fusion layer with rich metadata
        """
        processed_objects = []
        
        for obj in yolo_objects:
            confidence = obj.get('confidence', 0)
            if confidence >= self.object_confidence_threshold:
                processed_obj = {
                    "name": obj.get('name', obj.get('class', 'unknown')),
                    "confidence": confidence,
                    "bbox": obj.get('bbox', []),
                    "area": self._calculate_object_area(obj.get('bbox', [])),
                    "position": obj.get('position', 'unknown'),
                    "size_category": obj.get('size_category', 'unknown'),
                    "category": obj.get('category', 'other'),
                    "category_description": obj.get('category_description', ''),
                    "description": obj.get('description', ''),
                    "size_ratio": obj.get('size_ratio', 0),
                    "center_point": obj.get('center_point', [])
                }
                processed_objects.append(processed_obj)
        
        # Sort by confidence
        processed_objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"🔀 Processed {len(processed_objects)}/{len(yolo_objects)} enhanced objects with rich metadata")
        return processed_objects
    
    def _analyze_spatial_relationships(self, objects: List[Dict[str, Any]]) -> List[str]:
        """
        Analyze spatial relationships between objects
        """
        relationships = []
        
        if len(objects) < 2:
            return relationships
        
        # Group objects by position
        left_objects = [obj for obj in objects if obj.get('position') == 'left']
        center_objects = [obj for obj in objects if obj.get('position') == 'center']
        right_objects = [obj for obj in objects if obj.get('position') == 'right']
        
        # Generate spatial descriptions
        if left_objects and right_objects:
            relationships.append(f"{left_objects[0]['name']} on the left, {right_objects[0]['name']} on the right")
        
        if center_objects:
            relationships.append(f"{center_objects[0]['name']} in the center")
        
        # Find largest objects
        if objects:
            largest_obj = max(objects, key=lambda x: x.get('area', 0))
            if largest_obj['area'] > 0.2:  # If object takes up >20% of image
                relationships.append(f"prominent {largest_obj['name']}")
        
        # Count multiple instances
        object_counts = {}
        for obj in objects:
            name = obj['name']
            object_counts[name] = object_counts.get(name, 0) + 1
        
        multiple_objects = [name for name, count in object_counts.items() if count > 1]
        if multiple_objects:
            relationships.append(f"multiple {multiple_objects[0]}s" if len(multiple_objects) == 1 else "various repeated objects")
        
        return relationships[:3]  # Limit to top 3 relationships
    
    def _generate_scene_context(self, clip_keywords: List[str], blip_caption: str) -> Dict[str, Any]:
        """
        Generate scene context from CLIP keywords and BLIP caption
        """
        # Categorize keywords
        environment_keywords = ['indoor', 'outdoor', 'natural', 'artificial', 'urban', 'rural']
        lighting_keywords = ['bright', 'dark', 'sunny', 'shadowy', 'illuminated']
        mood_keywords = ['peaceful', 'busy', 'chaotic', 'serene', 'active', 'static']
        style_keywords = ['modern', 'vintage', 'classic', 'contemporary', 'traditional']
        
        context = {
            "environment": [kw for kw in clip_keywords if kw in environment_keywords],
            "lighting": [kw for kw in clip_keywords if kw in lighting_keywords],
            "mood": [kw for kw in clip_keywords if kw in mood_keywords],
            "style": [kw for kw in clip_keywords if kw in style_keywords],
            "characteristics": []
        }
        
        # Generate characteristics based on context
        if context["environment"]:
            context["characteristics"].append(f"{context['environment'][0]} setting")
        if context["lighting"]:
            context["characteristics"].append(f"{context['lighting'][0]} lighting")
        if context["mood"]:
            context["characteristics"].append(f"{context['mood'][0]} atmosphere")
        
        return context
    
    def _create_enhanced_description(
        self,
        blip_caption: str,
        objects: List[Dict[str, Any]],
        spatial_analysis: List[str],
        scene_context: Dict[str, Any],
        clip_keywords: List[str],
        fusion_mode: str,
        user_query: Optional[str] = None
    ) -> str:
        """
        Create the final enhanced description using fusion with rich object context
        """
        # Prepare components with enhanced object data
        object_list = self._format_enhanced_object_list(objects[:5])  # Top 5 objects with rich metadata
        main_objects = [obj['name'] for obj in objects[:3]]  # Top 3 objects
        spatial_context = ". ".join(spatial_analysis) if spatial_analysis else ""
        quality_context = ". ".join(scene_context.get('characteristics', []))
        
        # Get category summary for richer context
        category_summary = self._get_object_category_summary(objects)
        category_context = self._format_category_context(category_summary)
        
        # Choose template based on mode
        template = self.fusion_templates.get(fusion_mode, self.fusion_templates["detailed"])
        
        # Fill template with enhanced context
        try:
            if fusion_mode == "detailed":
                description = template.format(
                    blip_caption=blip_caption or "an image",
                    object_list=object_list or "various elements",
                    spatial_context=spatial_context,
                    quality_context=quality_context
                )
                # Add category context for detailed mode
                if category_context:
                    description += f" {category_context}"
                    
            elif fusion_mode == "concise":
                description = template.format(
                    blip_caption=blip_caption or "an image",
                    object_count=len(objects),
                    main_objects=", ".join(main_objects[:2]) if main_objects else "objects"
                )
                
            elif fusion_mode == "technical":
                detailed_objects = self._create_detailed_object_list(objects[:3])
                scene_analysis = ", ".join(clip_keywords[:5])
                description = template.format(
                    blip_caption=blip_caption or "visual content",
                    detailed_objects=detailed_objects,
                    scene_analysis=scene_analysis or "standard scene"
                )
            else:
                description = blip_caption or "Unable to generate description"
                
        except Exception as e:
            logger.warning(f"⚠️ Template formatting failed: {e}")
            description = f"{blip_caption}. {object_list}." if blip_caption and object_list else blip_caption or "Image analysis completed"
        
        # Post-process description
        description = self._post_process_description(description, user_query)
        
        return description
    
    def _format_object_list(self, objects: List[Dict[str, Any]]) -> str:
        """
        Format object list for natural language
        """
        if not objects:
            return ""
        
        object_names = [obj['name'] for obj in objects]
        
        if len(object_names) == 1:
            return f"a {object_names[0]}"
        elif len(object_names) == 2:
            return f"a {object_names[0]} and a {object_names[1]}"
        elif len(object_names) <= 5:
            return f"{', '.join(object_names[:-1])}, and a {object_names[-1]}"
        else:
            return f"{', '.join(object_names[:3])}, and {len(object_names)-3} other objects"
    
    def _format_enhanced_object_list(self, objects: List[Dict[str, Any]]) -> str:
        """
        Format object list with enhanced metadata for natural language
        """
        if not objects:
            return ""
        
        # Group objects by size category for better description
        dominant_objects = [obj for obj in objects if obj.get('size_category') == 'dominant']
        prominent_objects = [obj for obj in objects if obj.get('size_category') == 'prominent']
        other_objects = [obj for obj in objects if obj.get('size_category') not in ['dominant', 'prominent']]
        
        description_parts = []
        
        # Describe dominant objects first
        if dominant_objects:
            dominant_names = [obj['name'] for obj in dominant_objects]
            if len(dominant_names) == 1:
                description_parts.append(f"a dominant {dominant_names[0]}")
            else:
                description_parts.append(f"dominant elements including {', '.join(dominant_names)}")
        
        # Add prominent objects
        if prominent_objects:
            prominent_names = [obj['name'] for obj in prominent_objects]
            if len(prominent_names) == 1:
                description_parts.append(f"a prominent {prominent_names[0]}")
            elif len(prominent_names) <= 3:
                description_parts.append(f"prominent {', '.join(prominent_names)}")
            else:
                description_parts.append(f"several prominent objects including {', '.join(prominent_names[:2])}")
        
        # Add other objects if space
        if other_objects and len(description_parts) < 2:
            other_names = [obj['name'] for obj in other_objects[:2]]
            if other_names:
                description_parts.append(f"and {', '.join(other_names)}")
        
        return ", ".join(description_parts) if description_parts else "various objects"
    
    def _create_detailed_object_list(self, objects: List[Dict[str, Any]]) -> str:
        """
        Create detailed technical object list with enhanced metadata
        """
        if not objects:
            return "no objects detected"
        
        detailed_items = []
        for obj in objects:
            confidence = obj.get('confidence', 0)
            name = obj['name']
            size_cat = obj.get('size_category', 'unknown')
            category = obj.get('category', 'other')
            position = obj.get('position', 'unknown')
            
            # Create detailed description with all metadata
            detail = f"{name} ({confidence:.2f}, {size_cat}, {category}, {position})"
            detailed_items.append(detail)
        
        return "; ".join(detailed_items)
    
    def _post_process_description(self, description: str, user_query: Optional[str] = None) -> str:
        """
        Post-process the description for quality and relevance
        """
        # Clean up double spaces and redundant punctuation
        description = " ".join(description.split())
        description = description.replace("...", ".").replace(",,", ",")
        
        # Ensure proper capitalization
        if description and not description[0].isupper():
            description = description[0].upper() + description[1:]
        
        # Ensure proper ending
        if description and not description.endswith('.'):
            description += "."
        
        # Add query-specific context if relevant
        if user_query and len(user_query.strip()) > 3:
            query_words = user_query.lower().split()
            description_words = description.lower().split()
            
            # Check if query words appear in description
            query_coverage = sum(1 for word in query_words if word in description_words) / len(query_words)
            
            if query_coverage < 0.3:  # Low coverage, add context
                description += f" This relates to your question about {user_query.lower()}."
        
        return description
    
    def _calculate_object_area(self, bbox: List[float]) -> float:
        """
        Calculate normalized area of object bounding box
        """
        if len(bbox) < 4:
            return 0.0
        
        x1, y1, x2, y2 = bbox[:4]
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # Normalize assuming image coordinates are 0-1 or calculate percentage
        if max(x1, y1, x2, y2) <= 1.0:
            return width * height  # Already normalized
        else:
            # Assume pixel coordinates, normalize by typical image size
            return (width * height) / (640 * 640)  # Normalize by typical YOLO input size
    
    def _categorize_position(self, bbox: List[float]) -> str:
        """
        Categorize object position in image
        """
        if len(bbox) < 4:
            return "unknown"
        
        x1, y1, x2, y2 = bbox[:4]
        center_x = (x1 + x2) / 2
        
        # Normalize if needed
        if max(x1, x2) > 1.0:
            center_x = center_x / 640  # Assuming 640px typical width
        
        if center_x < 0.33:
            return "left"
        elif center_x > 0.67:
            return "right"
        else:
            return "center"
    
    def _categorize_object_size(self, area: float) -> str:
        """
        Categorize object size based on area
        """
        if area <= 0:
            return "tiny"
        elif area < 0.05:
            return "small"
        elif area < 0.15:
            return "medium"
        else:
            return "large"
    
    def _calculate_fusion_quality(
        self,
        blip_result: Dict[str, Any],
        objects: List[Dict[str, Any]],
        clip_keywords: List[str],
        enhanced_description: str
    ) -> float:
        """
        Calculate the quality of the fusion result with enhanced object scoring
        """
        quality_factors = []
        
        # BLIP caption quality
        blip_caption = blip_result.get('caption', '')
        if blip_caption and len(blip_caption.split()) >= 3:
            quality_factors.append(0.3)  # Good caption
        elif blip_caption:
            quality_factors.append(0.15)  # Minimal caption
        
        # Enhanced object detection quality with metadata scoring
        high_conf_objects = [obj for obj in objects if obj.get('confidence', 0) > 0.7]
        categorized_objects = [obj for obj in objects if obj.get('category') and obj.get('category') != 'other']
        dominant_objects = [obj for obj in objects if obj.get('size_category') == 'dominant']
        
        # Base object score
        if len(high_conf_objects) >= 3:
            quality_factors.append(0.25)
        elif len(high_conf_objects) >= 1:
            quality_factors.append(0.15)
        
        # Bonus for categorized objects (rich metadata)
        if len(categorized_objects) >= 2:
            quality_factors.append(0.1)
        elif len(categorized_objects) >= 1:
            quality_factors.append(0.05)
        
        # Bonus for dominant objects (size analysis)
        if dominant_objects:
            quality_factors.append(0.08)
        
        # CLIP keywords quality
        if len(clip_keywords) >= 5:
            quality_factors.append(0.2)
        elif len(clip_keywords) >= 2:
            quality_factors.append(0.1)
        
        # Enhanced description quality with metadata bonus
        if enhanced_description and len(enhanced_description.split()) >= 10:
            quality_factors.append(0.25)
        elif enhanced_description:
            quality_factors.append(0.1)
        
        # Bonus for rich metadata integration
        metadata_score = 0
        if any(obj.get('position') for obj in objects):
            metadata_score += 0.05  # Position data
        if any(obj.get('size_category') for obj in objects):
            metadata_score += 0.05  # Size categorization
        if any(obj.get('category_description') for obj in objects):
            metadata_score += 0.05  # Category descriptions
        
        quality_factors.append(metadata_score)
        
        return min(sum(quality_factors), 1.0)
    
    def _get_object_category_summary(self, objects: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Get summary of object categories detected
        """
        category_counts = {}
        for obj in objects:
            category = obj.get('category', 'other')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return category_counts
    
    def _format_category_context(self, category_summary: Dict[str, int]) -> str:
        """
        Format the category summary for inclusion in the description
        """
        if not category_summary:
            return ""
        
        # Sort categories by count
        sorted_categories = sorted(category_summary.items(), key=lambda x: x[1], reverse=True)
        
        # Create context string
        top_categories = [f"{cat} ({count})" for cat, count in sorted_categories[:3]]
        other_count = sum(count for cat, count in sorted_categories[3:])
        
        if other_count > 0:
            top_categories.append(f"and {other_count} other categories")
        
        return "Includes: " + ", ".join(top_categories)
    
    def _generate_recommendations(self, fusion_quality: float, metadata: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations for improving fusion quality
        """
        recommendations = []
        
        if fusion_quality < 0.5:
            recommendations.append("Consider using higher quality images for better analysis")
        
        component_count = metadata.get('component_count', {})
        if component_count.get('detected_objects', 0) < 2:
            recommendations.append("Object detection found few objects - try images with more distinct elements")
        
        if component_count.get('clip_keywords', 0) < 3:
            recommendations.append("Semantic analysis limited - try images with clearer scenes")
        
        if metadata.get('processing_time', 0) > 5.0:
            recommendations.append("Processing took longer than expected - consider resizing large images")
        
        if not recommendations:
            recommendations.append("Fusion quality is good - all components working well")
        
        return recommendations

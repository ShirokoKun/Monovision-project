"""
MonoVision V3 - Enhanced Chatbot System
Advanced conversational AI with precision vision & language interaction
"""
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import numpy as np

from ..core.orchestrator import MonoVisionOrchestrator, ProcessingMode, ProcessingRequest
from ..nlp.context_prompts import ContextAwarePrompts
from ..vision.fusion_layer import VisionFusionLayer

logger = logging.getLogger(__name__)

@dataclass
class ChatbotContext:
    """Enhanced chatbot context with session memory"""
    session_id: str
    user_preferences: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    current_image: Optional[Image.Image] = None
    last_vision_results: Optional[Dict[str, Any]] = None
    interaction_count: int = 0
    preferred_mode: Optional[str] = "balanced"

@dataclass
class IntelligentResponse:
    """Enhanced response with explanation and metadata"""
    text: str
    confidence: float
    mode_used: str
    reasoning: Dict[str, Any]
    suggestions: List[str]
    processing_time: float
    tokens_used: int
    vision_insights: Optional[Dict[str, Any]] = None

class EnhancedChatbotSystem:
    """
    V3 Enhanced Chatbot with Advanced Features:
    - Precision Vision & Language Interaction
    - Dynamic Object & Attribute Queries
    - Contextual Analysis and Explanations
    - Intelligent Mode Recommendations
    - Real-time Performance Feedback
    """
    
    def __init__(self, orchestrator: MonoVisionOrchestrator):
        self.orchestrator = orchestrator
        self.context_prompts = ContextAwarePrompts()
        self.active_sessions: Dict[str, ChatbotContext] = {}
        
        # Enhanced chatbot capabilities
        self.vision_query_patterns = {
            'object_specific': [
                r'what (?:is|are) the (.+?) (?:in|at|on) the (.+?)\?',
                r'describe the (.+?) (?:in|at|on) the (.+)',
                r'tell me about the (.+?) (?:person|object|item)'
            ],
            'mood_analysis': [
                r'what (?:is|are) the (?:mood|feeling|atmosphere)',
                r'how (?:does|do) (?:this|the) (?:look|feel|seem)',
                r'what (?:emotions?|vibes?) (?:do you see|are present)'
            ],
            'spatial_relationships': [
                r'where (?:is|are) the (.+?)\?',
                r'what (?:is|are) (?:next to|behind|in front of|above|below) the (.+?)\?',
                r'how (?:is|are) the (.+?) positioned'
            ],
            'comparison_queries': [
                r'compare the (.+?) (?:and|with) the (.+)',
                r'what (?:is|are) the difference(?:s?) between (.+?) and (.+)',
                r'which (?:is|are) (?:bigger|smaller|brighter|darker)'
            ],
            'explanation_requests': [
                r'why did you say (.+?)\?',
                r'explain (?:why|how) (.+)',
                r'what makes you think (.+?)\?'
            ]
        }
        
        logger.info("🤖 Enhanced Chatbot System initialized")
    
    async def process_intelligent_query(
        self, 
        query: str, 
        image: Optional[Image.Image] = None,
        session_id: str = "default",
        preferred_mode: Optional[str] = None
    ) -> IntelligentResponse:
        """
        Process user query with intelligent analysis and mode selection
        """
        # Get or create session context
        context = self._get_session_context(session_id)
        context.interaction_count += 1
        
        # Update current image if provided
        if image:
            context.current_image = image
        
        # Intelligent mode recommendation
        recommended_mode = self._recommend_processing_mode(query, context, preferred_mode)
        
        # Create processing request
        request = ProcessingRequest(
            image_data=self._image_to_bytes(context.current_image) if context.current_image else b'',
            query=query,
            mode=ProcessingMode(recommended_mode),
            session_id=session_id,
            request_id=f"chat_{session_id}_{context.interaction_count:04d}",
            include_objects=True
        )
        
        # Process with orchestrator
        response = await self.orchestrator.process_request(request)
        
        # Create intelligent response with enhancements
        intelligent_response = await self._create_intelligent_response(
            query, response, context, recommended_mode
        )
        
        # Update conversation history
        self._update_conversation_history(context, query, intelligent_response)
        
        return intelligent_response
    
    def _recommend_processing_mode(
        self, 
        query: str, 
        context: ChatbotContext, 
        preferred_mode: Optional[str]
    ) -> str:
        """
        Intelligent mode recommendation based on query analysis
        """
        if preferred_mode:
            return preferred_mode
        
        query_lower = query.lower()
        
        # Rich mode indicators
        rich_indicators = [
            'detailed analysis', 'comprehensive', 'artistic', 'cultural context',
            'expert opinion', 'professional analysis', 'in-depth', 'thorough',
            'composition', 'lighting', 'technique', 'style analysis'
        ]
        
        # Fast mode indicators
        fast_indicators = [
            'quick', 'brief', 'simple', 'what is', 'identify', 'name',
            'yes or no', 'true or false', 'count', 'how many'
        ]
        
        # Balanced mode indicators (spatial, object-specific)
        balanced_indicators = [
            'where', 'position', 'location', 'spatial', 'relationship',
            'mood', 'atmosphere', 'feeling', 'emotion', 'describe',
            'explain', 'analyze', 'tell me about'
        ]
        
        # Check for specific pattern matches
        if any(indicator in query_lower for indicator in rich_indicators):
            return "rich"
        elif any(indicator in query_lower for indicator in fast_indicators):
            return "fast"
        elif any(indicator in query_lower for indicator in balanced_indicators):
            return "balanced"
        
        # Default to balanced for most comprehensive analysis
        return context.preferred_mode or "balanced"
    
    async def _create_intelligent_response(
        self,
        query: str,
        orchestrator_response,
        context: ChatbotContext,
        mode_used: str
    ) -> IntelligentResponse:
        """
        Create enhanced response with reasoning and suggestions
        """
        # Extract vision insights if available
        vision_insights = None
        if orchestrator_response.vision_results:
            vision_insights = {
                'objects_detected': len(orchestrator_response.vision_results.get('objects', [])),
                'semantic_keywords': orchestrator_response.vision_results.get('clip_keywords', []),
                'fusion_quality': orchestrator_response.vision_results.get('fusion_quality', 0.0),
                'scene_complexity': self._analyze_scene_complexity(orchestrator_response.vision_results)
            }
        
        # Generate reasoning
        reasoning = self._generate_reasoning(query, orchestrator_response, mode_used)
        
        # Generate intelligent suggestions
        suggestions = await self._generate_suggestions(query, orchestrator_response, context)
        
        # Calculate confidence based on fusion quality and response coherence
        confidence = self._calculate_response_confidence(orchestrator_response, vision_insights)
        
        return IntelligentResponse(
            text=orchestrator_response.language_response,
            confidence=confidence,
            mode_used=mode_used,
            reasoning=reasoning,
            suggestions=suggestions,
            processing_time=orchestrator_response.processing_time,
            tokens_used=orchestrator_response.token_count,
            vision_insights=vision_insights
        )
    
    def _generate_reasoning(
        self, 
        query: str, 
        response, 
        mode_used: str
    ) -> Dict[str, Any]:
        """
        Generate explanation for why specific analysis was performed
        """
        reasoning = {
            'mode_selection': f"Used {mode_used} mode for optimal balance of detail and speed",
            'analysis_focus': [],
            'confidence_factors': []
        }
        
        # Add mode-specific reasoning
        if mode_used == "balanced":
            reasoning['mode_selection'] = "Balanced mode selected for comprehensive object detection and semantic analysis"
            reasoning['analysis_focus'].extend([
                "Enhanced object detection (15 objects max, spatial analysis)",
                "Advanced semantic analysis (70 keywords across 6 categories)",
                "Extended responses (up to 100 tokens)"
            ])
        elif mode_used == "rich":
            reasoning['mode_selection'] = "Rich mode selected for expert-level analysis"
            reasoning['analysis_focus'].extend([
                "Professional-grade analysis with cultural context",
                "Comprehensive 150-token responses",
                "Advanced artistic and technical insights"
            ])
        else:  # fast
            reasoning['mode_selection'] = "Fast mode selected for quick, efficient responses"
            reasoning['analysis_focus'].extend([
                "Rapid processing (3-8 seconds)",
                "Essential information extraction",
                "Concise 25-token responses"
            ])
        
        # Add confidence factors based on vision results
        if hasattr(response, 'vision_results') and response.vision_results:
            fusion_quality = response.vision_results.get('fusion_quality', 0.0)
            if fusion_quality > 0.8:
                reasoning['confidence_factors'].append("High fusion quality (>0.8)")
            if response.vision_results.get('objects'):
                reasoning['confidence_factors'].append(f"Detected {len(response.vision_results['objects'])} objects")
            if response.vision_results.get('clip_keywords'):
                reasoning['confidence_factors'].append(f"Found {len(response.vision_results['clip_keywords'])} semantic keywords")
        
        return reasoning
    
    async def _generate_suggestions(
        self,
        query: str,
        response,
        context: ChatbotContext
    ) -> List[str]:
        """
        Generate intelligent follow-up suggestions
        """
        suggestions = []
        
        # Base suggestions based on vision results
        if hasattr(response, 'vision_results') and response.vision_results:
            objects = response.vision_results.get('objects', [])
            keywords = response.vision_results.get('clip_keywords', [])
            
            if objects:
                suggestions.append(f"Ask about specific objects: 'Tell me more about the {objects[0]['name'] if isinstance(objects[0], dict) else objects[0]}'")
            
            if 'group' in keywords or 'social' in keywords:
                suggestions.append("Explore social context: 'What's the mood or atmosphere here?'")
            
            if any(style_word in keywords for style_word in ['modern', 'vintage', 'artistic']):
                suggestions.append("Analyze style: 'What artistic style or technique is used?'")
            
            if len(objects) > 3:
                suggestions.append("Ask about spatial relationships: 'How are the objects arranged in the scene?'")
        
        # Mode-specific suggestions
        current_mode = response.mode if hasattr(response, 'mode') else 'balanced'
        if current_mode == 'fast':
            suggestions.append("Try Balanced mode for more detailed analysis")
        elif current_mode == 'balanced':
            suggestions.append("Try Rich mode for professional-level insights")
        
        # Query-pattern specific suggestions
        query_lower = query.lower()
        if 'what' in query_lower and context.current_image:
            suggestions.append("Ask 'why' questions: 'Why does this image have that particular mood?'")
        
        if 'person' in query_lower or 'people' in query_lower:
            suggestions.append("Explore emotions: 'What emotions or feelings do you see?'")
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    def _analyze_scene_complexity(self, vision_results: Dict[str, Any]) -> str:
        """
        Analyze scene complexity based on vision results
        """
        objects_count = len(vision_results.get('objects', []))
        keywords_count = len(vision_results.get('clip_keywords', []))
        
        if objects_count >= 8 or keywords_count >= 5:
            return "high"
        elif objects_count >= 4 or keywords_count >= 3:
            return "medium"
        else:
            return "low"
    
    def _calculate_response_confidence(
        self, 
        response, 
        vision_insights: Optional[Dict[str, Any]]
    ) -> float:
        """
        Calculate confidence score based on multiple factors
        """
        confidence = 0.7  # Base confidence
        
        if vision_insights:
            # Fusion quality factor
            fusion_quality = vision_insights.get('fusion_quality', 0.0)
            confidence += (fusion_quality - 0.5) * 0.3
            
            # Object detection factor
            objects_count = vision_insights.get('objects_detected', 0)
            if objects_count > 0:
                confidence += min(objects_count * 0.05, 0.15)
            
            # Semantic keywords factor
            keywords_count = len(vision_insights.get('semantic_keywords', []))
            if keywords_count > 0:
                confidence += min(keywords_count * 0.02, 0.1)
        
        # Token count factor (good responses tend to be substantive)
        if hasattr(response, 'token_count') and response.token_count > 10:
            confidence += 0.05
        
        return min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1
    
    def _get_session_context(self, session_id: str) -> ChatbotContext:
        """Get or create session context"""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = ChatbotContext(
                session_id=session_id,
                user_preferences={},
                conversation_history=[],
                interaction_count=0,
                preferred_mode="balanced"
            )
        return self.active_sessions[session_id]
    
    def _update_conversation_history(
        self, 
        context: ChatbotContext, 
        query: str, 
        response: IntelligentResponse
    ):
        """Update conversation history"""
        context.conversation_history.append({
            'timestamp': asyncio.get_event_loop().time(),
            'query': query,
            'response': response.text,
            'mode_used': response.mode_used,
            'confidence': response.confidence,
            'processing_time': response.processing_time,
            'tokens_used': response.tokens_used
        })
        
        # Keep only last 20 interactions
        if len(context.conversation_history) > 20:
            context.conversation_history = context.conversation_history[-20:]
    
    def _image_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to bytes"""
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        return buffer.getvalue()
    
    async def explain_response(
        self, 
        session_id: str, 
        aspect: str = "reasoning"
    ) -> str:
        """
        Explain why the chatbot gave a particular response
        """
        context = self._get_session_context(session_id)
        
        if not context.conversation_history:
            return "No previous responses to explain."
        
        last_interaction = context.conversation_history[-1]
        
        if aspect == "reasoning":
            return f"""I analyzed your image using {last_interaction['mode_used']} mode because it's optimal for your type of query. 
            
My confidence was {last_interaction['confidence']:.2f} based on:
- Vision analysis quality
- Number of objects detected
- Semantic keyword matches
- Response coherence

Processing took {last_interaction['processing_time']:.2f}s and generated {last_interaction['tokens_used']} tokens."""
        
        elif aspect == "mode_selection":
            return f"""I chose {last_interaction['mode_used']} mode for your query because:

- **Fast Mode (25 tokens, ~3-8s)**: Best for quick identification tasks
- **Balanced Mode (100 tokens, ~15-30s)**: Optimal for detailed analysis with spatial intelligence  
- **Rich Mode (150 tokens, ~30-60s)**: Professional-grade analysis with cultural context

Your query pattern suggested {last_interaction['mode_used']} mode would provide the best balance of detail and efficiency."""
        
        return "I can explain my reasoning, mode selection, or confidence factors. What would you like to know?"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get enhanced chatbot system status"""
        return {
            'active_sessions': len(self.active_sessions),
            'total_interactions': sum(ctx.interaction_count for ctx in self.active_sessions.values()),
            'average_confidence': self._calculate_average_confidence(),
            'mode_usage_stats': self._get_mode_usage_stats(),
            'capabilities': {
                'precision_vision_queries': True,
                'spatial_relationship_analysis': True,
                'mood_and_atmosphere_detection': True,
                'intelligent_mode_recommendation': True,
                'conversation_context_memory': True,
                'explanation_and_reasoning': True
            }
        }
    
    def _calculate_average_confidence(self) -> float:
        """Calculate average confidence across all sessions"""
        all_confidences = []
        for context in self.active_sessions.values():
            for interaction in context.conversation_history:
                if 'confidence' in interaction:
                    all_confidences.append(interaction['confidence'])
        
        return sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    
    def _get_mode_usage_stats(self) -> Dict[str, int]:
        """Get statistics on mode usage"""
        mode_counts = {'fast': 0, 'balanced': 0, 'rich': 0}
        
        for context in self.active_sessions.values():
            for interaction in context.conversation_history:
                mode = interaction.get('mode_used', 'balanced')
                if mode in mode_counts:
                    mode_counts[mode] += 1
        
        return mode_counts

class ObjectSpecificQueryProcessor:
    """
    Advanced processor for object-specific queries with spatial intelligence
    """
    
    def __init__(self, vision_fusion: VisionFusionLayer):
        self.vision_fusion = vision_fusion
    
    async def process_object_query(
        self, 
        image: Image.Image, 
        target_object: str, 
        location_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process queries about specific objects in specific locations
        """
        # Get full vision analysis
        vision_results = await self.vision_fusion.analyze_image(image)
        
        # Find matching objects
        matching_objects = self._find_matching_objects(
            vision_results.get('objects', []), 
            target_object, 
            location_hint
        )
        
        if not matching_objects:
            return {
                'found': False,
                'message': f"Could not find '{target_object}' in the image",
                'suggestions': self._suggest_visible_objects(vision_results.get('objects', []))
            }
        
        # Analyze specific objects
        object_analysis = []
        for obj in matching_objects:
            analysis = await self._analyze_specific_object(image, obj, vision_results)
            object_analysis.append(analysis)
        
        return {
            'found': True,
            'target_object': target_object,
            'matching_objects': matching_objects,
            'detailed_analysis': object_analysis,
            'spatial_context': self._analyze_spatial_context(matching_objects, vision_results.get('objects', []))
        }
    
    def _find_matching_objects(
        self, 
        objects: List[Dict[str, Any]], 
        target: str, 
        location_hint: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Find objects matching the target description"""
        matches = []
        target_lower = target.lower()
        
        for obj in objects:
            obj_name = obj.get('name', '').lower()
            
            # Direct name match
            if target_lower in obj_name or obj_name in target_lower:
                matches.append(obj)
            
            # Semantic matching for common synonyms
            elif self._semantic_object_match(target_lower, obj_name):
                matches.append(obj)
        
        # If location hint provided, filter by position
        if location_hint and matches:
            matches = self._filter_by_location(matches, location_hint)
        
        return matches
    
    def _semantic_object_match(self, target: str, obj_name: str) -> bool:
        """Check for semantic matches between object names"""
        synonyms = {
            'person': ['human', 'man', 'woman', 'people', 'individual'],
            'car': ['vehicle', 'automobile', 'auto'],
            'building': ['house', 'structure', 'construction'],
            'animal': ['pet', 'creature', 'wildlife']
        }
        
        for key, syn_list in synonyms.items():
            if (target in syn_list and obj_name == key) or (obj_name in syn_list and target == key):
                return True
        
        return False
    
    def _filter_by_location(
        self, 
        objects: List[Dict[str, Any]], 
        location_hint: str
    ) -> List[Dict[str, Any]]:
        """Filter objects by spatial location hints"""
        location_lower = location_hint.lower()
        
        if 'center' in location_lower or 'middle' in location_lower:
            # Find objects closest to center
            return sorted(objects, key=lambda obj: self._distance_from_center(obj))[:1]
        elif 'left' in location_lower:
            return [obj for obj in objects if self._is_on_left(obj)]
        elif 'right' in location_lower:
            return [obj for obj in objects if self._is_on_right(obj)]
        elif 'top' in location_lower or 'upper' in location_lower:
            return [obj for obj in objects if self._is_on_top(obj)]
        elif 'bottom' in location_lower or 'lower' in location_lower:
            return [obj for obj in objects if self._is_on_bottom(obj)]
        
        return objects
    
    def _distance_from_center(self, obj: Dict[str, Any]) -> float:
        """Calculate distance from image center"""
        if 'bbox' not in obj:
            return float('inf')
        
        bbox = obj['bbox']
        obj_center_x = (bbox[0] + bbox[2]) / 2
        obj_center_y = (bbox[1] + bbox[3]) / 2
        
        # Assuming image dimensions are normalized to [0, 1] or we can get them
        image_center_x, image_center_y = 0.5, 0.5
        
        return ((obj_center_x - image_center_x) ** 2 + (obj_center_y - image_center_y) ** 2) ** 0.5
    
    def _is_on_left(self, obj: Dict[str, Any]) -> bool:
        """Check if object is on the left side"""
        if 'bbox' not in obj:
            return False
        return (obj['bbox'][0] + obj['bbox'][2]) / 2 < 0.5
    
    def _is_on_right(self, obj: Dict[str, Any]) -> bool:
        """Check if object is on the right side"""
        if 'bbox' not in obj:
            return False
        return (obj['bbox'][0] + obj['bbox'][2]) / 2 > 0.5
    
    def _is_on_top(self, obj: Dict[str, Any]) -> bool:
        """Check if object is on the top half"""
        if 'bbox' not in obj:
            return False
        return (obj['bbox'][1] + obj['bbox'][3]) / 2 < 0.5
    
    def _is_on_bottom(self, obj: Dict[str, Any]) -> bool:
        """Check if object is on the bottom half"""
        if 'bbox' not in obj:
            return False
        return (obj['bbox'][1] + obj['bbox'][3]) / 2 > 0.5
    
    async def _analyze_specific_object(
        self, 
        image: Image.Image, 
        obj: Dict[str, Any], 
        vision_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform detailed analysis of a specific object"""
        analysis = {
            'object_name': obj.get('name', 'unknown'),
            'confidence': obj.get('confidence', 0.0),
            'position': self._describe_position(obj),
            'size_analysis': self._analyze_object_size(obj, vision_results.get('objects', [])),
            'context': self._analyze_object_context(obj, vision_results)
        }
        
        return analysis
    
    def _describe_position(self, obj: Dict[str, Any]) -> str:
        """Generate human-readable position description"""
        if 'bbox' not in obj:
            return "Position unknown"
        
        bbox = obj['bbox']
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        horizontal = "center"
        if center_x < 0.33:
            horizontal = "left"
        elif center_x > 0.67:
            horizontal = "right"
        
        vertical = "middle"
        if center_y < 0.33:
            vertical = "top"
        elif center_y > 0.67:
            vertical = "bottom"
        
        if horizontal == "center" and vertical == "middle":
            return "center of the image"
        elif horizontal == "center":
            return f"{vertical} center"
        elif vertical == "middle":
            return f"{horizontal} side"
        else:
            return f"{vertical} {horizontal}"
    
    def _analyze_object_size(
        self, 
        obj: Dict[str, Any], 
        all_objects: List[Dict[str, Any]]
    ) -> str:
        """Analyze object size relative to other objects"""
        if 'bbox' not in obj:
            return "Size unknown"
        
        obj_area = self._calculate_bbox_area(obj['bbox'])
        
        if len(all_objects) < 2:
            return f"Occupies {obj_area:.1%} of the image"
        
        # Compare with other objects
        other_areas = [
            self._calculate_bbox_area(other_obj['bbox']) 
            for other_obj in all_objects 
            if 'bbox' in other_obj and other_obj != obj
        ]
        
        if obj_area > max(other_areas):
            return f"Largest object ({obj_area:.1%} of image)"
        elif obj_area < min(other_areas):
            return f"Smallest object ({obj_area:.1%} of image)"
        else:
            return f"Medium-sized object ({obj_area:.1%} of image)"
    
    def _calculate_bbox_area(self, bbox: List[float]) -> float:
        """Calculate normalized area of bounding box"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return width * height
    
    def _analyze_object_context(
        self, 
        obj: Dict[str, Any], 
        vision_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze object within its visual context"""
        context = {
            'surrounding_objects': self._find_nearby_objects(obj, vision_results.get('objects', [])),
            'semantic_context': self._get_semantic_context(obj, vision_results.get('clip_keywords', [])),
            'scene_role': self._determine_scene_role(obj, vision_results)
        }
        
        return context
    
    def _find_nearby_objects(
        self, 
        target_obj: Dict[str, Any], 
        all_objects: List[Dict[str, Any]]
    ) -> List[str]:
        """Find objects spatially close to the target object"""
        if 'bbox' not in target_obj:
            return []
        
        nearby = []
        target_bbox = target_obj['bbox']
        
        for obj in all_objects:
            if obj == target_obj or 'bbox' not in obj:
                continue
            
            # Calculate spatial proximity
            if self._are_objects_nearby(target_bbox, obj['bbox']):
                nearby.append(obj.get('name', 'unknown'))
        
        return nearby[:3]  # Limit to 3 nearby objects
    
    def _are_objects_nearby(self, bbox1: List[float], bbox2: List[float], threshold: float = 0.3) -> bool:
        """Check if two objects are spatially nearby"""
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2
        
        distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        return distance < threshold
    
    def _get_semantic_context(
        self, 
        obj: Dict[str, Any], 
        clip_keywords: List[str]
    ) -> List[str]:
        """Get relevant semantic keywords for the object"""
        # Filter keywords that might relate to the object
        relevant_keywords = []
        obj_name = obj.get('name', '').lower()
        
        if 'person' in obj_name:
            relevant_keywords.extend([kw for kw in clip_keywords if kw in ['social', 'formal', 'casual', 'professional', 'group']])
        elif obj_name in ['car', 'vehicle']:
            relevant_keywords.extend([kw for kw in clip_keywords if kw in ['urban', 'street', 'modern', 'vintage']])
        
        return relevant_keywords[:3]
    
    def _determine_scene_role(
        self, 
        obj: Dict[str, Any], 
        vision_results: Dict[str, Any]
    ) -> str:
        """Determine the object's role in the overall scene"""
        obj_area = self._calculate_bbox_area(obj.get('bbox', [0, 0, 0, 0]))
        
        if obj_area > 0.3:
            return "dominant element"
        elif obj_area > 0.1:
            return "significant element"
        elif obj.get('confidence', 0) > 0.8:
            return "clear focal point"
        else:
            return "background element"
    
    def _suggest_visible_objects(self, objects: List[Dict[str, Any]]) -> List[str]:
        """Suggest objects that are actually visible in the image"""
        visible_objects = [obj.get('name', 'unknown') for obj in objects if obj.get('confidence', 0) > 0.5]
        return list(set(visible_objects))[:5]  # Unique objects, max 5
    
    def _analyze_spatial_context(
        self, 
        target_objects: List[Dict[str, Any]], 
        all_objects: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze spatial relationships between target objects and the scene"""
        if not target_objects:
            return {}
        
        spatial_context = {
            'object_count': len(target_objects),
            'scene_distribution': self._analyze_scene_distribution(target_objects),
            'relative_positions': self._analyze_relative_positions(target_objects, all_objects),
            'clustering': self._analyze_object_clustering(target_objects)
        }
        
        return spatial_context
    
    def _analyze_scene_distribution(self, objects: List[Dict[str, Any]]) -> str:
        """Analyze how objects are distributed across the scene"""
        if not objects:
            return "No objects"
        
        positions = []
        for obj in objects:
            if 'bbox' in obj:
                center_x = (obj['bbox'][0] + obj['bbox'][2]) / 2
                center_y = (obj['bbox'][1] + obj['bbox'][3]) / 2
                positions.append((center_x, center_y))
        
        if len(positions) == 1:
            return "Single object"
        
        # Calculate spread
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        x_spread = max(x_coords) - min(x_coords)
        y_spread = max(y_coords) - min(y_coords)
        
        if x_spread > 0.5 and y_spread > 0.5:
            return "Spread across entire scene"
        elif x_spread > 0.5:
            return "Spread horizontally"
        elif y_spread > 0.5:
            return "Spread vertically"
        else:
            return "Clustered together"
    
    def _analyze_relative_positions(
        self, 
        target_objects: List[Dict[str, Any]], 
        all_objects: List[Dict[str, Any]]
    ) -> List[str]:
        """Analyze positions relative to other scene elements"""
        relationships = []
        
        for target_obj in target_objects:
            if 'bbox' not in target_obj:
                continue
            
            # Find relationships with other objects
            for other_obj in all_objects:
                if other_obj in target_objects or 'bbox' not in other_obj:
                    continue
                
                relationship = self._describe_spatial_relationship(
                    target_obj['bbox'], 
                    other_obj['bbox'],
                    target_obj.get('name', 'object'),
                    other_obj.get('name', 'object')
                )
                if relationship:
                    relationships.append(relationship)
        
        return relationships[:3]  # Limit to 3 most important relationships
    
    def _describe_spatial_relationship(
        self, 
        bbox1: List[float], 
        bbox2: List[float], 
        name1: str, 
        name2: str
    ) -> Optional[str]:
        """Describe spatial relationship between two objects"""
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2
        
        dx = center1_x - center2_x
        dy = center1_y - center2_y
        
        if abs(dx) > abs(dy):
            if dx > 0.2:
                return f"{name1} is to the right of {name2}"
            elif dx < -0.2:
                return f"{name1} is to the left of {name2}"
        else:
            if dy > 0.2:
                return f"{name1} is below {name2}"
            elif dy < -0.2:
                return f"{name1} is above {name2}"
        
        # If objects are close, check for overlap or adjacency
        distance = (dx**2 + dy**2)**0.5
        if distance < 0.3:
            return f"{name1} is near {name2}"
        
        return None
    
    def _analyze_object_clustering(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze clustering patterns of objects"""
        if len(objects) < 2:
            return {"pattern": "single object"}
        
        positions = []
        for obj in objects:
            if 'bbox' in obj:
                center_x = (obj['bbox'][0] + obj['bbox'][2]) / 2
                center_y = (obj['bbox'][1] + obj['bbox'][3]) / 2
                positions.append((center_x, center_y))
        
        if len(positions) < 2:
            return {"pattern": "insufficient data"}
        
        # Calculate average inter-object distance
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = ((positions[i][0] - positions[j][0])**2 + 
                       (positions[i][1] - positions[j][1])**2)**0.5
                distances.append(dist)
        
        avg_distance = sum(distances) / len(distances)
        
        clustering_analysis = {
            "pattern": "clustered" if avg_distance < 0.3 else "distributed",
            "average_separation": avg_distance,
            "total_objects": len(objects)
        }
        
        return clustering_analysis

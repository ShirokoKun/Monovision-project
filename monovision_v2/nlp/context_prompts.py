"""
Context-Aware Response Templates for MonoVision V3
Dynamic prompt generation based on processing mode and vision content
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ContextPrompt:
    """Template for context-aware prompts"""
    system_prompt: str
    user_template: str
    vision_template: str
    object_template: str
    style_guidelines: str

class ContextAwarePrompts:
    """Generates dynamic prompts based on processing mode and vision content"""
    
    def __init__(self):
        self.prompts = {
            "fast": ContextPrompt(
                system_prompt="""You are an efficient AI assistant providing quick, accurate responses about images. 
Be concise, factual, and focus on the most important elements. Keep responses under 50 words unless specifically asked for more detail.""",
                
                user_template="""Quick analysis needed: {user_input}""",
                
                vision_template="""Image shows: {caption}
Key elements: {clip_keywords}
Context: {user_input}
Provide a brief, focused response.""",
                
                object_template="""Objects detected: {objects}
Image shows: {caption}
Key elements: {clip_keywords}
Context: {user_input}
Provide a brief, focused response.""",
                
                style_guidelines="Concise, direct, factual tone. Prioritize key information."
            ),
            
            "balanced": ContextPrompt(
                system_prompt="""You are a knowledgeable AI assistant providing thoughtful, well-rounded responses about images. 
Balance detail with clarity. Provide comprehensive but accessible explanations. Aim for detailed responses when analyzing images with multiple elements. Focus on spatial relationships, composition, and contextual meaning.""",
                
                user_template="""Detailed analysis: {user_input}""",
                
                vision_template="""Image Analysis:
- Caption: {caption}
- Visual characteristics: {clip_keywords}
- User query: {user_input}

Provide a thorough yet accessible response that addresses the visual content and user's specific question. Include details about composition, spatial relationships, and contextual meaning where relevant.""",
                
                object_template="""Enhanced Image Analysis:
- Caption: {caption}
- Objects present: {objects}
- Visual characteristics: {clip_keywords}
- User query: {user_input}

Provide a comprehensive analysis that incorporates object detection insights with the overall visual understanding. Discuss spatial relationships, object interactions, and the overall composition. Address the user's question with detailed context.""",
                
                style_guidelines="Informative, balanced detail level with enhanced depth. Connect visual elements to user queries meaningfully and discuss spatial relationships."
            ),
            
            "rich": ContextPrompt(
                system_prompt="""You are an expert AI assistant providing in-depth, nuanced responses about images. 
Offer detailed analysis, explore implications, and provide educational context. Use your full knowledge to create rich, informative responses with deep insights into visual composition, cultural context, and artistic elements.""",
                
                user_template="""Comprehensive analysis requested: {user_input}""",
                
                vision_template="""Comprehensive Image Analysis:
- Primary content: {caption}
- Visual semantics: {clip_keywords}
- User inquiry: {user_input}

Provide an in-depth response that explores the visual content thoroughly, offers contextual insights, discusses composition techniques, and addresses any educational or analytical aspects relevant to the user's question. Include observations about lighting, color theory, perspective, and cultural context where applicable.""",
                
                object_template="""Expert Image Analysis:
- Primary content: {caption}
- Detected objects: {objects}
- Visual semantics: {clip_keywords}  
- User inquiry: {user_input}

Provide an expert-level analysis that integrates object detection with overall scene understanding. Explore relationships between detected objects, discuss potential contexts or scenarios, analyze spatial composition, examine visual hierarchy, and offer educational insights relevant to the visual content and user's question. Consider artistic techniques, cultural significance, and compositional elements.""",
                
                style_guidelines="Expert-level detail with comprehensive analysis, educational context, thorough exploration of visual relationships, artistic techniques, and cultural implications."
            )
        }
    
    def generate_context_prompt(
        self, 
        mode: str, 
        user_input: str, 
        vision_results: Dict[str, Any],
        enable_fusion: bool = True,
        max_tokens: int = 100,
        session_memory: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, str]:
        """
        Generate context-aware prompt based on processing mode and vision content
        
        Args:
            mode: Processing mode ("fast", "balanced", "rich")
            user_input: User's text input/question
            vision_results: Results from vision fusion layer
            enable_fusion: Whether to use enhanced fusion processing
            max_tokens: Maximum tokens for the response
            session_memory: Previous conversation history for context
            
        Returns:
            Dict with system_prompt, user_prompt, and additional metadata
        """
        try:
            if mode not in self.prompts:
                mode = "balanced"  # Default fallback
                
            prompt_template = self.prompts[mode]
            
            # Extract vision data
            caption = vision_results.get("caption", "")
            # Handle both string and dict formats for clip_keywords
            raw_keywords = vision_results.get("clip_keywords", [])
            if raw_keywords and isinstance(raw_keywords[0], dict):
                clip_keywords = ", ".join([str(kw.get('name', kw.get('keyword', str(kw)))) for kw in raw_keywords])
            else:
                clip_keywords = ", ".join([str(kw) for kw in raw_keywords])
            objects = vision_results.get("objects", [])
            
            # Check if this is text-only mode (no image provided)
            is_text_only = not caption and not clip_keywords and not objects
            
            # Handle fusion results if available and fusion is enabled
            fusion_result = vision_results.get("fusion_result", {})
            if enable_fusion and fusion_result:
                # Use enhanced fusion description if available
                enhanced_caption = fusion_result.get("enhanced_description", caption)
                fusion_quality = fusion_result.get("fusion_quality", 0.0)
            else:
                enhanced_caption = caption
                fusion_quality = 0.0
            
            # Build context based on mode
            if is_text_only:
                # Text-only mode: use simple user template without vision context
                logger.info(f"💬 Text-only mode detected for {mode} processing")
                vision_context = prompt_template.user_template.format(user_input=user_input)
            elif objects and len(objects) > 0:
                # Use object-aware template - FIX: Handle YOLO object dicts properly
                if isinstance(objects[0], dict):
                    # YOLO format: [{"name": "person", "confidence": 0.8, "bbox": [...]}]
                    objects_str = ", ".join([obj["name"] for obj in objects[:5]])  # Top 5 objects
                else:
                    # String format: ["person", "car", "tree"]
                    objects_str = ", ".join(objects[:5])  # Top 5 objects
                vision_context = prompt_template.object_template.format(
                    caption=enhanced_caption,
                    objects=objects_str,
                    clip_keywords=clip_keywords,
                    user_input=user_input
                )
            else:
                # Use standard vision template
                vision_context = prompt_template.vision_template.format(
                    caption=enhanced_caption,
                    clip_keywords=clip_keywords,
                    user_input=user_input
                )
            
            # Generate final prompts
            system_prompt = prompt_template.system_prompt
            
            # Add session memory context if available
            if session_memory and len(session_memory) > 0:
                memory_context = "\n\nPrevious conversation context:\n"
                for i, mem in enumerate(session_memory[:3]):  # Use last 3 interactions
                    memory_context += f"Q{i+1}: {mem['query']}\nA{i+1}: {mem['response']}\n"
                
                # Enhance system prompt with conversation context
                system_prompt += f"\n\nYou have access to previous conversation history. Use this context to provide more relevant and coherent responses, but focus primarily on the current question."
                user_prompt = memory_context + "\nCurrent question: " + vision_context
            else:
                user_prompt = vision_context
            
            # Log prompt generation (fixed f-string syntax)
            if is_text_only:
                detail_msg = "(text-only)"
            else:
                num_keywords = len(clip_keywords.split(",")) if clip_keywords else 0
                detail_msg = f"with {len(objects)} objects, {num_keywords} keywords"
            logger.info(f"📝 Generated {mode} mode prompt {detail_msg}")
            
            # Enhanced return format with metadata (V3 compatible)
            return {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "mode": mode,
                "style_guidelines": prompt_template.style_guidelines,
                "context_metadata": {
                    "template_used": "text_only" if is_text_only else ("object_template" if objects else "vision_template"),
                    "fusion_enabled": enable_fusion and not is_text_only,
                    "token_limit": max_tokens,
                    "objects_count": len(objects),
                    "keywords_count": len(clip_keywords.split(',')) if clip_keywords else 0,
                    "is_text_only": is_text_only
                },
                "vision_summary": {
                    "fusion_quality": fusion_quality,
                    "enhanced_caption": enhanced_caption,
                    "original_caption": caption if caption else "No image provided"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Context prompt generation error: {e}")
            # Fallback to simple prompt with same enhanced format
            return {
                "system_prompt": "You are a helpful AI assistant analyzing images.",
                "user_prompt": f"Image: {vision_results.get('caption', 'No caption available')}\nUser: {user_input}\nPlease provide a helpful response.",
                "mode": mode,
                "style_guidelines": "Be helpful and informative.",
                "context_metadata": {
                    "template_used": "fallback",
                    "fusion_enabled": False,
                    "token_limit": max_tokens,
                    "objects_count": 0,
                    "keywords_count": 0
                },
                "vision_summary": {
                    "fusion_quality": 0.0,
                    "enhanced_caption": "Error in processing",
                    "original_caption": vision_results.get('caption', 'No caption available')
                }
            }
    
    def get_mode_description(self, mode: str) -> str:
        """Get description of processing mode capabilities"""
        descriptions = {
            "fast": "Quick responses, BLIP caption only, optimized for speed",
            "balanced": "Detailed analysis with BLIP + CLIP + optional YOLOv8, balanced speed/quality", 
            "rich": "Comprehensive analysis with all models, maximum detail and context"
        }
        return descriptions.get(mode, "Unknown mode")
    
    def get_enhanced_system_context(self, mode: str) -> str:
        """Get enhanced system context for better AI responses"""
        if mode not in self.prompts:
            mode = "balanced"
            
        return f"""
Context-Aware Processing Mode: {mode.upper()}
{self.get_mode_description(mode)}

Style Guidelines: {self.prompts[mode].style_guidelines}

Remember to:
1. Adapt response length and detail to the processing mode
2. Integrate visual elements meaningfully with user queries  
3. Maintain consistency with the mode's intended user experience
4. Use detected objects and visual keywords to enhance understanding
"""

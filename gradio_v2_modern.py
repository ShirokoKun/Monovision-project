"""
MonoVision V4 - Ultra-Enhanced AI Vision Interface
Advanced object detection with improved accuracy and comprehensive image analysis
Modern VS Code inspired dark UI with enhanced features and detailed insights

🎯 Key Enhancements:
• Enhanced Object Detection: Improved accuracy with confidence filtering (>35%), size validation, and duplicate removal
• Comprehensive Image Analysis: Color temperature, composition rules, style detection, lighting analysis, texture patterns, and emotional impact assessment
• Advanced Filtering Pipeline: Multi-stage object detection with relevance checking and position categorization
• Modular Analysis Architecture: Specialized functions for different aspects of image analysis
• Enhanced UI Theme: Brighter, more visible colors for better accessibility and user experience

🔧 Technical Features:
• Confidence-based filtering with minimum 35% threshold
• Size validation to eliminate tiny/irrelevant detections
• Duplicate object removal using bounding box overlap detection
• Color analysis with KMeans clustering for dominant colors
• Composition analysis with rule of thirds and balance detection
• Style and mood detection based on visual patterns
• Scene classification for context understanding
• Lighting analysis for brightness and contrast assessment
• Texture analysis for surface pattern complexity
• Emotional impact assessment based on visual cues
"""

import os
import time
import logging
import asyncio
import gradio as gr
from typing import Optional, Tuple, List, Dict, Any
import json
import hashlib
from PIL import Image
import base64
import io
import subprocess
import socket
import random
import numpy as np
from collections import Counter

from monovision_v2 import (
    MonoVisionOrchestrator,
    ProcessingMode,
    ProcessingRequest,
    ProcessingResponse
)
from monovision_v2.ui.enhanced_ui_components import ObjectOverlayGenerator, SmartImageInteraction

# Import enhanced V3 components if available
try:
    from monovision_v2.ui.enhanced_chatbot import EnhancedChatbotSystem, ObjectSpecificQueryProcessor
    from monovision_v2.ui.production_dashboard import ProductionDashboard, IntelligentModeSelector
    ENHANCED_FEATURES_AVAILABLE = True
    print("✅ Enhanced V3 features loaded successfully")
except ImportError as e:
    print(f"⚠️ Enhanced features not available: {e}")
    print("🔄 Falling back to standard V2 features")
    ENHANCED_FEATURES_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🌑 Enhanced Modern Dark Theme Colors
THEME_COLORS = {
   "background": "#0B0B0E",          # Rich near-black (slightly blue-tinted for depth)
"surface": "#16161C",             # Smooth dark gray for panels (less harsh than pure black)
"primary": "#00BFFF",             # Electric blue (pops but less eye-fatiguing than #00D4FF)
"secondary": "#FF5C93",           # Softer neon pink (still vibrant, more balanced with blue)
"accent": "#32FF7E",              # Fresh mint-neon green (less toxic, still glows)
"text_primary": "#F5F5F7",        # Off-white for softer readability
"text_secondary": "#A5A5B0",      # Muted gray with slight blue tint
"text_muted": "#6E6E73",          # Subtle dark gray for low hierarchy
"border": "#2C2C34",              # Softer contrast borders
"success": "#21E6A1",             # Calming emerald green
"warning": "#FFC857",             # Modern amber-yellow
"error": "#FF4D6D",               # Rose-red (less harsh than bright red)
"info": "#3BA7FF"                 # Calm sky blue for info
            
}

# Enhanced Object Detection Configuration
OBJECT_DETECTION_CONFIG = {
    "min_confidence": 0.35,          # Higher confidence threshold
    "min_size_ratio": 0.01,          # Minimum object size (1% of image)
    "max_objects": 15,               # Maximum objects to detect
    "relevant_classes": [            # Focus on these object types
        "person", "car", "truck", "bus", "motorcycle", "bicycle",
        "dog", "cat", "bird", "horse", "sheep", "cow",
        "chair", "table", "sofa", "bed", "tv", "laptop", "mouse", "keyboard", "cell phone",
        "bottle", "cup", "fork", "knife", "spoon", "bowl",
        "apple", "banana", "orange", "broccoli", "carrot",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
}

# Enhanced Image Analysis Configuration
IMAGE_ANALYSIS_CONFIG = {
    "color_analysis": True,
    "composition_analysis": True,
    "style_detection": True,
    "scene_classification": True,
    "lighting_analysis": True,
    "texture_analysis": True,
    "emotion_detection": True
}

def enhanced_object_detection(objects: List[Dict[str, Any]], image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
    """
    Enhanced object detection with improved accuracy and filtering

    Args:
        objects: Raw detected objects from YOLO
        image_size: (width, height) of the original image

    Returns:
        Filtered and enhanced object list
    """
    if not objects:
        return []

    filtered_objects = []
    image_area = image_size[0] * image_size[1]

    for obj in objects:
        # Extract object properties
        confidence = obj.get('confidence', 0)
        class_name = obj.get('name', obj.get('class', 'unknown'))
        bbox = obj.get('bbox', obj.get('bounding_box', []))

        # Confidence filtering
        if confidence < OBJECT_DETECTION_CONFIG["min_confidence"]:
            continue

        # Size filtering - skip very small objects
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox
            obj_width = abs(x2 - x1)
            obj_height = abs(y2 - y1)
            obj_area = obj_width * obj_height

            # Skip if object is too small relative to image
            if obj_area / image_area < OBJECT_DETECTION_CONFIG["min_size_ratio"]:
                continue

            # Skip if object dimensions are unreasonable
            if obj_width < 5 or obj_height < 5:
                continue

        # Class relevance filtering
        if class_name.lower() not in [c.lower() for c in OBJECT_DETECTION_CONFIG["relevant_classes"]]:
            # Allow some generic classes but with higher confidence requirement
            if confidence < 0.5:
                continue

        # Enhanced object information
        enhanced_obj = obj.copy()
        enhanced_obj.update({
            'confidence_percent': round(confidence * 100, 1),
            'size_category': _categorize_object_size(obj_area / image_area if bbox else 0),
            'position': _get_object_position(bbox, image_size) if bbox else 'unknown',
            'description': _get_object_description(class_name, confidence)
        })

        filtered_objects.append(enhanced_obj)

    # Remove duplicates and sort by confidence
    filtered_objects = _remove_duplicate_objects(filtered_objects)
    filtered_objects = sorted(filtered_objects, key=lambda x: x.get('confidence', 0), reverse=True)

    # Limit number of objects
    return filtered_objects[:OBJECT_DETECTION_CONFIG["max_objects"]]

def _categorize_object_size(size_ratio: float) -> str:
    """Categorize object size based on relative area"""
    if size_ratio > 0.3:
        return "large"
    elif size_ratio > 0.1:
        return "medium"
    elif size_ratio > 0.02:
        return "small"
    else:
        return "tiny"

def _get_object_position(bbox: List[float], image_size: Tuple[int, int]) -> str:
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

def _get_object_description(class_name: str, confidence: float) -> str:
    """Generate descriptive text for detected object"""
    confidence_text = "high" if confidence > 0.7 else "medium" if confidence > 0.5 else "low"
    return f"{class_name} (confidence: {confidence_text})"

def _remove_duplicate_objects(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate objects based on class and position similarity"""
    if not objects:
        return objects

    unique_objects = []
    for obj in objects:
        is_duplicate = False
        obj_class = obj.get('name', obj.get('class', ''))
        obj_bbox = obj.get('bbox', obj.get('bounding_box', []))

        for existing in unique_objects:
            existing_class = existing.get('name', existing.get('class', ''))
            existing_bbox = existing.get('bbox', existing.get('bounding_box', []))

            # Same class and overlapping bounding boxes
            if (obj_class == existing_class and
                _calculate_bbox_overlap(obj_bbox, existing_bbox) > 0.5):
                is_duplicate = True
                break

        if not is_duplicate:
            unique_objects.append(obj)

    return unique_objects

def _calculate_bbox_overlap(bbox1: List[float], bbox2: List[float]) -> float:
    """Calculate overlap ratio between two bounding boxes"""
    if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
        return 0.0

    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def enhanced_image_analysis(image: Image.Image, vision_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform comprehensive image analysis beyond basic caption and objects

    Args:
        image: PIL Image object
        vision_results: Basic vision analysis results

    Returns:
        Enhanced analysis dictionary
    """
    analysis = {
        'color_analysis': {},
        'composition': {},
        'style_mood': {},
        'scene_type': {},
        'lighting': {},
        'texture': {},
        'emotional_impact': {}
    }

    try:
        # Convert to numpy array for analysis
        img_array = np.array(image)

        # Color Analysis
        if IMAGE_ANALYSIS_CONFIG["color_analysis"]:
            analysis['color_analysis'] = _analyze_image_colors(img_array)

        # Composition Analysis
        if IMAGE_ANALYSIS_CONFIG["composition_analysis"]:
            analysis['composition'] = _analyze_composition(img_array, vision_results)

        # Style and Mood Detection
        if IMAGE_ANALYSIS_CONFIG["style_detection"]:
            analysis['style_mood'] = _analyze_style_and_mood(img_array, vision_results)

        # Scene Classification
        if IMAGE_ANALYSIS_CONFIG["scene_classification"]:
            analysis['scene_type'] = _classify_scene(vision_results)

        # Lighting Analysis
        if IMAGE_ANALYSIS_CONFIG["lighting_analysis"]:
            analysis['lighting'] = _analyze_lighting(img_array)

        # Texture Analysis
        if IMAGE_ANALYSIS_CONFIG["texture_analysis"]:
            analysis['texture'] = _analyze_texture(img_array)

        # Emotional Impact
        if IMAGE_ANALYSIS_CONFIG["emotion_detection"]:
            analysis['emotional_impact'] = _analyze_emotional_impact(vision_results, analysis)

    except Exception as e:
        logger.warning(f"Enhanced image analysis failed: {e}")
        analysis['error'] = str(e)

    return analysis

def _analyze_image_colors(img_array: np.ndarray) -> Dict[str, Any]:
    """Analyze dominant colors and color scheme"""
    try:
        # Reshape image for color analysis
        pixels = img_array.reshape(-1, 3)

        # Calculate dominant colors using k-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
        kmeans.fit(pixels)

        # Get dominant colors
        dominant_colors = []
        for center in kmeans.cluster_centers_:
            color_hex = '#{:02x}{:02x}{:02x}'.format(
                int(center[0]), int(center[1]), int(center[2])
            )
            dominant_colors.append(color_hex)

        # Color temperature analysis
        avg_color = np.mean(pixels, axis=0)
        warmth = (avg_color[0] + avg_color[1]) / 2 - avg_color[2]  # Red + Green - Blue

        if warmth > 50:
            temperature = "warm"
        elif warmth < -50:
            temperature = "cool"
        else:
            temperature = "neutral"

        # Saturation analysis
        hsv = np.array(Image.fromarray(img_array).convert('HSV'))
        saturation = np.mean(hsv[:, :, 1])

        if saturation > 100:
            saturation_level = "vibrant"
        elif saturation > 50:
            saturation_level = "moderate"
        else:
            saturation_level = "muted"

        return {
            'dominant_colors': dominant_colors[:3],
            'temperature': temperature,
            'saturation': saturation_level,
            'brightness': "bright" if np.mean(img_array) > 128 else "dark"
        }

    except Exception as e:
        return {'error': f'Color analysis failed: {e}'}

def _analyze_composition(img_array: np.ndarray, vision_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze image composition and layout"""
    try:
        height, width = img_array.shape[:2]

        # Rule of thirds analysis
        third_h = height // 3
        third_w = width // 3

        # Check for objects at rule of thirds intersections
        rule_of_thirds_score = 0
        objects = vision_results.get('objects', [])

        for obj in objects:
            bbox = obj.get('bbox', [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Check if center is near rule of thirds lines
                if (abs(center_x - third_w) < third_w * 0.1 or
                    abs(center_x - 2 * third_w) < third_w * 0.1 or
                    abs(center_y - third_h) < third_h * 0.1 or
                    abs(center_y - 2 * third_h) < third_h * 0.1):
                    rule_of_thirds_score += 1

        # Aspect ratio analysis
        aspect_ratio = width / height
        if aspect_ratio > 1.5:
            orientation = "landscape"
        elif aspect_ratio < 0.67:
            orientation = "portrait"
        else:
            orientation = "square-ish"

        # Object distribution
        object_positions = []
        for obj in objects:
            bbox = obj.get('bbox', [])
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                object_positions.append((center_x / width, center_y / height))

        if object_positions:
            # Calculate center of mass
            avg_x = np.mean([pos[0] for pos in object_positions])
            avg_y = np.mean([pos[1] for pos in object_positions])

            if avg_x < 0.4:
                balance = "left-heavy"
            elif avg_x > 0.6:
                balance = "right-heavy"
            else:
                balance = "balanced"
        else:
            balance = "unknown"

        return {
            'orientation': orientation,
            'balance': balance,
            'rule_of_thirds_alignment': rule_of_thirds_score,
            'object_count': len(objects),
            'composition_type': _classify_composition_type(objects, (width, height))
        }

    except Exception as e:
        return {'error': f'Composition analysis failed: {e}'}

def _classify_composition_type(objects: List[Dict[str, Any]], image_size: Tuple[int, int]) -> str:
    """Classify the overall composition type"""
    if not objects:
        return "minimalist"

    # Analyze object sizes and positions
    large_objects = 0
    small_objects = 0

    for obj in objects:
        bbox = obj.get('bbox', [])
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            img_area = image_size[0] * image_size[1]

            if area / img_area > 0.1:
                large_objects += 1
            elif area / img_area < 0.02:
                small_objects += 1

    if large_objects >= 2:
        return "multi-subject"
    elif large_objects == 1:
        return "single-subject"
    elif small_objects > len(objects) * 0.7:
        return "detailed-scene"
    else:
        return "balanced-scene"

def _analyze_style_and_mood(img_array: np.ndarray, vision_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze artistic style and emotional mood"""
    try:
        # Analyze based on keywords and colors
        keywords = vision_results.get('clip_keywords', [])
        caption = vision_results.get('caption', '').lower()

        # Style detection
        style_indicators = {
            'photorealistic': ['photo', 'photograph', 'realistic', 'detailed'],
            'artistic': ['painting', 'art', 'illustration', 'drawing', 'sketch'],
            'vintage': ['vintage', 'retro', 'old', 'classic', 'antique'],
            'modern': ['modern', 'contemporary', 'sleek', 'minimal'],
            'natural': ['nature', 'outdoor', 'landscape', 'organic']
        }

        detected_styles = []
        for style, indicators in style_indicators.items():
            if any(indicator in caption or indicator in ' '.join(keywords) for indicator in indicators):
                detected_styles.append(style)

        # Mood detection
        mood_indicators = {
            'peaceful': ['peaceful', 'calm', 'serene', 'tranquil', 'quiet'],
            'energetic': ['energetic', 'dynamic', 'active', 'vibrant', 'lively'],
            'mysterious': ['mysterious', 'dark', 'shadowy', 'moody', 'intriguing'],
            'joyful': ['joyful', 'happy', 'bright', 'cheerful', 'colorful'],
            'melancholic': ['melancholic', 'sad', 'gloomy', 'somber', 'gray']
        }

        detected_moods = []
        for mood, indicators in mood_indicators.items():
            if any(indicator in caption or indicator in ' '.join(keywords) for indicator in indicators):
                detected_moods.append(mood)

        # Color-based mood inference
        avg_brightness = np.mean(img_array)
        color_std = np.std(img_array)

        if avg_brightness > 180:
            detected_moods.append('bright')
        elif avg_brightness < 80:
            detected_moods.append('dark')

        if color_std > 60:
            detected_moods.append('vibrant')
        elif color_std < 30:
            detected_moods.append('muted')

        return {
            'detected_styles': list(set(detected_styles)) or ['contemporary'],
            'detected_moods': list(set(detected_moods)) or ['neutral'],
            'primary_style': detected_styles[0] if detected_styles else 'contemporary',
            'primary_mood': detected_moods[0] if detected_moods else 'neutral'
        }

    except Exception as e:
        return {'error': f'Style/mood analysis failed: {e}'}

def _classify_scene(vision_results: Dict[str, Any]) -> Dict[str, Any]:
    """Classify the type of scene depicted"""
    try:
        keywords = vision_results.get('clip_keywords', [])
        caption = vision_results.get('caption', '').lower()
        objects = vision_results.get('objects', [])

        # Scene type classification
        scene_types = {
            'indoor': ['indoor', 'inside', 'room', 'house', 'building', 'kitchen', 'living room'],
            'outdoor': ['outdoor', 'outside', 'nature', 'park', 'street', 'garden', 'mountain'],
            'urban': ['city', 'urban', 'street', 'building', 'car', 'traffic', 'crowd'],
            'natural': ['nature', 'landscape', 'mountain', 'forest', 'ocean', 'sky', 'tree'],
            'portrait': ['person', 'face', 'portrait', 'selfie', 'people'],
            'food': ['food', 'dish', 'meal', 'fruit', 'vegetable', 'drink'],
            'animal': ['animal', 'dog', 'cat', 'bird', 'horse', 'pet'],
            'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle']
        }

        detected_scenes = []
        confidence_scores = {}

        for scene_type, indicators in scene_types.items():
            score = 0
            for indicator in indicators:
                if indicator in caption:
                    score += 3  # Higher weight for caption matches
                if any(indicator in keyword.lower() for keyword in keywords):
                    score += 2  # Medium weight for keyword matches
                if any(indicator in obj.get('name', '').lower() for obj in objects):
                    score += 1  # Lower weight for object matches

            if score > 0:
                detected_scenes.append(scene_type)
                confidence_scores[scene_type] = score

        # Sort by confidence
        detected_scenes.sort(key=lambda x: confidence_scores.get(x, 0), reverse=True)

        return {
            'primary_scene': detected_scenes[0] if detected_scenes else 'general',
            'detected_scenes': detected_scenes[:3],
            'scene_confidence': confidence_scores.get(detected_scenes[0], 0) if detected_scenes else 0
        }

    except Exception as e:
        return {'error': f'Scene classification failed: {e}'}

def _analyze_lighting(img_array: np.ndarray) -> Dict[str, Any]:
    """Analyze lighting conditions in the image"""
    try:
        # Convert to grayscale for lighting analysis
        gray = np.mean(img_array, axis=2).astype(np.uint8)

        # Overall brightness
        avg_brightness = np.mean(gray)

        # Contrast analysis
        contrast = np.std(gray)

        # Lighting distribution
        hist, _ = np.histogram(gray, bins=10, range=(0, 255))
        hist = hist / np.sum(hist)  # Normalize

        # Check for high contrast (bright and dark areas)
        high_brightness_ratio = np.sum(hist[7:])  # Top 30% brightness
        low_brightness_ratio = np.sum(hist[:3])   # Bottom 30% brightness

        if contrast > 60:
            lighting_type = "high_contrast"
        elif contrast > 30:
            lighting_type = "moderate_contrast"
        else:
            lighting_type = "low_contrast"

        # Direction inference based on shadows/highlights
        if high_brightness_ratio > 0.6:
            direction = "bright_even"
        elif low_brightness_ratio > 0.6:
            direction = "dark_even"
        else:
            direction = "directional"

        return {
            'overall_brightness': "bright" if avg_brightness > 150 else "dark" if avg_brightness < 100 else "medium",
            'contrast_level': lighting_type,
            'lighting_direction': direction,
            'brightness_score': round(avg_brightness / 255 * 100, 1)
        }

    except Exception as e:
        return {'error': f'Lighting analysis failed: {e}'}

def _analyze_texture(img_array: np.ndarray) -> Dict[str, Any]:
    """Analyze texture patterns in the image"""
    try:
        # Simple texture analysis using edge detection
        gray = np.mean(img_array, axis=2).astype(np.uint8)

        # Calculate gradients for texture detection
        dx = np.abs(np.gradient(gray, axis=1))
        dy = np.abs(np.gradient(gray, axis=0))

        # Edge magnitude
        edges = np.sqrt(dx**2 + dy**2)
        edge_density = np.mean(edges > 20)  # Percentage of strong edges

        # Texture complexity based on edge variation
        edge_std = np.std(edges)

        if edge_density > 0.3:
            texture_type = "detailed"
        elif edge_density > 0.1:
            texture_type = "moderate"
        else:
            texture_type = "smooth"

        return {
            'texture_type': texture_type,
            'edge_density': round(edge_density * 100, 1),
            'complexity_score': round(edge_std, 1),
            'surface_type': _infer_surface_type(edges, gray)
        }

    except Exception as e:
        return {'error': f'Texture analysis failed: {e}'}

def _infer_surface_type(edges: np.ndarray, gray: np.ndarray) -> str:
    """Infer the dominant surface type based on texture"""
    try:
        # Simple heuristics for surface type
        edge_mean = np.mean(edges)
        gray_std = np.std(gray)

        if edge_mean > 30 and gray_std > 40:
            return "rough_uneven"
        elif edge_mean > 20:
            return "textured"
        elif gray_std > 30:
            return "varied_smooth"
        else:
            return "smooth_uniform"

    except:
        return "unknown"

def _analyze_emotional_impact(vision_results: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the emotional impact of the image"""
    try:
        keywords = vision_results.get('clip_keywords', [])
        caption = vision_results.get('caption', '').lower()

        # Emotional keywords
        positive_words = ['happy', 'joyful', 'beautiful', 'peaceful', 'love', 'bright', 'colorful']
        negative_words = ['dark', 'sad', 'angry', 'scary', 'lonely', 'gloomy', 'gray']
        neutral_words = ['calm', 'serene', 'quiet', 'simple', 'clean']

        positive_score = sum(1 for word in positive_words if word in caption or word in keywords)
        negative_score = sum(1 for word in negative_words if word in caption or word in keywords)
        neutral_score = sum(1 for word in neutral_words if word in caption or word in keywords)

        # Color-based emotion
        color_analysis = analysis.get('color_analysis', {})
        if color_analysis.get('saturation') == 'vibrant':
            positive_score += 1
        if color_analysis.get('brightness') == 'dark':
            negative_score += 1

        # Determine dominant emotion
        max_score = max(positive_score, negative_score, neutral_score)
        if max_score == 0:
            dominant_emotion = 'neutral'
        elif positive_score == max_score:
            dominant_emotion = 'positive'
        elif negative_score == max_score:
            dominant_emotion = 'negative'
        else:
            dominant_emotion = 'calm'

        return {
            'dominant_emotion': dominant_emotion,
            'emotional_intensity': 'strong' if max_score >= 3 else 'moderate' if max_score >= 2 else 'subtle',
            'emotional_keywords': [word for word in positive_words + negative_words + neutral_words
                                 if word in caption or word in keywords]
        }

    except Exception as e:
        return {'error': f'Emotional analysis failed: {e}'}

# 🎨 Custom CSS for Modern Dark Theme
CUSTOM_CSS = f"""
/* Global Dark Theme */
.gradio-container {{
    background: {THEME_COLORS["background"]} !important;
    color: {THEME_COLORS["text_primary"]} !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}}

/* Main Interface Layout */
.app-container {{
    background: {THEME_COLORS["background"]};
    min-height: 100vh;
}}

/* Left Sidebar Styling */
.sidebar {{
    background: {THEME_COLORS["surface"]} !important;
    border-right: 1px solid {THEME_COLORS["border"]};
    padding: 20px;
    border-radius: 0;
}}

.sidebar .logo {{
    color: {THEME_COLORS["primary"]};
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 30px;
    text-align: center;
}}

/* Tier Selector Buttons */
.tier-selector {{
    margin: 20px 0;
}}

.tier-button {{
    background: {THEME_COLORS["surface"]} !important;
    border: 1px solid {THEME_COLORS["border"]} !important;
    color: {THEME_COLORS["text_primary"]} !important;
    border-radius: 8px !important;
    margin: 5px 0 !important;
    padding: 12px 16px !important;
    transition: all 0.2s ease !important;
}}

.tier-button:hover {{
    border-color: {THEME_COLORS["primary"]} !important;
    background: {THEME_COLORS["primary"]}20 !important;
}}

.tier-button.selected {{
    background: {THEME_COLORS["primary"]} !important;
    border-color: {THEME_COLORS["primary"]} !important;
    color: white !important;
}}

/* Image Upload Zone */
.upload-zone {{
    background: {THEME_COLORS["surface"]} !important;
    border: 2px dashed {THEME_COLORS["border"]} !important;
    border-radius: 12px !important;
    padding: 30px !important;
    text-align: center !important;
    color: {THEME_COLORS["text_secondary"]} !important;
    transition: all 0.3s ease !important;
}}

.upload-zone:hover {{
    border-color: {THEME_COLORS["primary"]} !important;
    background: {THEME_COLORS["primary"]}10 !important;
}}

/* Main Workspace */
.workspace {{
    background: {THEME_COLORS["background"]};
    padding: 20px;
}}

/* Image Card */
.image-card {{
    background: {THEME_COLORS["surface"]} !important;
    border: 1px solid {THEME_COLORS["border"]} !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin: 10px 0 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;
}}

/* Chat Interface - Enhanced with Streaming Effects */
.chat-container {{
    background: {THEME_COLORS["background"]};
    border-radius: 12px;
    max-height: 600px;
    overflow-y: auto;
}}

.chat-message {{
    background: {THEME_COLORS["surface"]} !important;
    border: 1px solid {THEME_COLORS["border"]} !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin: 10px 0 !important;
    position: relative !important;
    animation: fadeInUp 0.3s ease-out;
}}

/* Streaming Animation for AI Messages */
.chat-message.ai {{
    border-left: 4px solid {THEME_COLORS["primary"]} !important;
    margin-right: 20px !important;
}}

.chat-message.ai:last-child {{
    animation: typewriter 0.1s linear;
}}

/* Typing Effect Animation */
@keyframes typewriter {{
    from {{
        opacity: 0.7;
    }}
    to {{
        opacity: 1;
    }}
}}

/* Fade In Animation for New Messages */
@keyframes fadeInUp {{
    from {{
        opacity: 0;
        transform: translateY(10px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

.chat-message.user {{
    border-left: 4px solid {THEME_COLORS["text_secondary"]} !important;
    margin-left: 20px !important;
}}

/* Processing Indicator */
.processing-indicator {{
    background: {THEME_COLORS["primary"]}20 !important;
    border: 1px solid {THEME_COLORS["primary"]} !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin: 10px 20px 10px 0 !important;
    position: relative !important;
    overflow: hidden !important;
}}

.processing-indicator::before {{
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, {THEME_COLORS["primary"]}40, transparent);
    animation: shimmer 2s infinite;
}}

@keyframes shimmer {{
    0% {{
        left: -100%;
    }}
    100% {{
        left: 100%;
    }}
}}

/* Cursor Effect for Streaming Text */
.streaming-text::after {{
    content: '|';
    color: {THEME_COLORS["primary"]};
    animation: blink 1s infinite;
    font-weight: bold;
}}

@keyframes blink {{
    0%, 50% {{
        opacity: 1;
    }}
    51%, 100% {{
        opacity: 0;
    }}
}}

/* Image preview in chat */
.chat-image {{
    max-width: 300px !important;
    max-height: 200px !important;
    border-radius: 8px !important;
    border: 1px solid {THEME_COLORS["border"]} !important;
    margin: 10px 0 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
}}

.image-preview-container {{
    background: {THEME_COLORS["surface"]} !important;
    border: 1px solid {THEME_COLORS["border"]} !important;
    border-radius: 8px !important;
    padding: 10px !important;
    margin: 10px 0 !important;
    text-align: center !important;
}}

/* Enhanced performance metrics */
.performance-badge {{
    display: inline-block !important;
    background: {THEME_COLORS["primary"]}20 !important;
    color: {THEME_COLORS["primary"]} !important;
    padding: 4px 8px !important;
    border-radius: 12px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    margin: 2px 4px !important;
}}

.cache-badge {{
    background: {THEME_COLORS["success"]}20 !important;
    color: {THEME_COLORS["success"]} !important;
}}

.time-badge {{
    background: {THEME_COLORS["warning"]}20 !important;
    color: {THEME_COLORS["warning"]} !important;
}}

/* Input Fields */
.input-field {{
    background: {THEME_COLORS["surface"]} !important;
    border: 1px solid {THEME_COLORS["border"]} !important;
    border-radius: 8px !important;
    color: {THEME_COLORS["text_primary"]} !important;
    padding: 12px 16px !important;
}}

.input-field:focus {{
    border-color: {THEME_COLORS["primary"]} !important;
    box-shadow: 0 0 0 2px {THEME_COLORS["primary"]}30 !important;
    outline: none !important;
}}

/* Buttons with Enhanced Hover Effects */
.btn-primary {{
    background: {THEME_COLORS["primary"]} !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    position: relative !important;
    overflow: hidden !important;
}}

.btn-primary:hover {{
    background: {THEME_COLORS["primary"]}CC !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px {THEME_COLORS["primary"]}50 !important;
}}

.btn-primary:active {{
    transform: translateY(0) !important;
    box-shadow: 0 2px 6px {THEME_COLORS["primary"]}30 !important;
}}

.btn-secondary {{
    background: {THEME_COLORS["surface"]} !important;
    border: 1px solid {THEME_COLORS["border"]} !important;
    color: {THEME_COLORS["text_primary"]} !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    transition: all 0.2s ease !important;
}}

.btn-secondary:hover {{
    border-color: {THEME_COLORS["primary"]} !important;
    background: {THEME_COLORS["primary"]}10 !important;
}}

/* Status Bar */
.status-bar {{
    background: {THEME_COLORS["surface"]}CC !important;
    border-bottom: 1px solid {THEME_COLORS["border"]} !important;
    padding: 8px 20px !important;
    font-size: 12px !important;
    color: {THEME_COLORS["text_secondary"]} !important;
}}

/* Performance Metrics */
.metrics {{
    background: {THEME_COLORS["surface"]} !important;
    border: 1px solid {THEME_COLORS["border"]} !important;
    border-radius: 8px !important;
    padding: 12px !important;
    margin: 10px 0 !important;
}}

.metric-item {{
    color: {THEME_COLORS["text_secondary"]} !important;
    font-size: 13px !important;
    margin: 2px 0 !important;
}}

/* Scrollbars */
::-webkit-scrollbar {{
    width: 8px;
}}

::-webkit-scrollbar-track {{
    background: {THEME_COLORS["surface"]};
}}

::-webkit-scrollbar-thumb {{
    background: {THEME_COLORS["border"]};
    border-radius: 4px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: {THEME_COLORS["primary"]};
}}

/* Override Gradio defaults */
.gradio-container .prose {{
    color: {THEME_COLORS["text_primary"]} !important;
}}

.gradio-container .prose h1, 
.gradio-container .prose h2, 
.gradio-container .prose h3 {{
    color: {THEME_COLORS["text_primary"]} !important;
}}

/* File upload area */
.file-upload {{
    background: {THEME_COLORS["surface"]} !important;
    border: 2px dashed {THEME_COLORS["border"]} !important;
    border-radius: 12px !important;
}}

.file-upload:hover {{
    border-color: {THEME_COLORS["primary"]} !important;
}}

/* Markdown rendering */
.markdown-content {{
    background: {THEME_COLORS["surface"]} !important;
    border-radius: 8px !important;
    padding: 16px !important;
    border: 1px solid {THEME_COLORS["border"]} !important;
}}

.markdown-content pre {{
    background: {THEME_COLORS["background"]} !important;
    border: 1px solid {THEME_COLORS["border"]} !important;
    border-radius: 6px !important;
    padding: 12px !important;
}}

.markdown-content code {{
    background: {THEME_COLORS["background"]} !important;
    color: {THEME_COLORS["primary"]} !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
}}

/* Enhanced V3 Components Styling */

/* Image Preview Container */
.image-preview-container {{
    background: {THEME_COLORS["surface"]};
    border: 1px solid {THEME_COLORS["border"]};
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}}

.chat-image {{
    max-width: 100%;
    max-height: 400px;
    border-radius: 8px;
    border: 2px solid {THEME_COLORS["border"]};
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}}

/* Object Overlay Styling */
.object-overlay-container {{
    position: relative;
    display: inline-block;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}}

.object-box {{
    transition: all 0.2s ease;
    z-index: 10;
}}

.object-box:hover {{
    transform: scale(1.02);
    z-index: 20;
    border-width: 3px !important;
}}

/* Performance Panel Enhanced */
.performance-panel-v3 {{
    background: linear-gradient(135deg, {THEME_COLORS["surface"]}, #252525);
    border: 1px solid {THEME_COLORS["border"]};
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
}}

.performance-panel-v3:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.4);
    border-color: {THEME_COLORS["primary"]};
}}

/* Cache Panel Styling */
.cache-panel-v3 {{
    background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
    border: 1px solid {THEME_COLORS["border"]};
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
    transition: all 0.3s ease;
}}

.cache-panel-v3:hover {{
    border-color: {THEME_COLORS["secondary"]};
}}

/* Fusion Comparison */
.fusion-comparison-v3 {{
    background: linear-gradient(135deg, {THEME_COLORS["surface"]}, #2d2d2d);
    border: 1px solid {THEME_COLORS["border"]};
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
    transition: all 0.3s ease;
}}

.fusion-comparison-v3:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.3);
}}

/* Async Processing Animation */
.async-processing-indicator {{
    margin: 8px 0;
    animation: pulse 2s infinite;
}}

@keyframes pulse {{
    0% {{ opacity: 0.8; }}
    50% {{ opacity: 1; }}
    100% {{ opacity: 0.8; }}
}}

/* Metric Items */
.metric-item {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}}

.metric-item:last-child {{
    border-bottom: none;
}}

/* Responsive Design */
@media (max-width: 768px) {{
    .performance-panel-v3 > div:first-child {{
        flex-direction: column !important;
        align-items: flex-start !important;
    }}
    
    .metrics-grid {{
        grid-template-columns: 1fr !important;
    }}
    
    .fusion-comparison-v3 > div {{
        grid-template-columns: 1fr !important;
    }}
    
    .chat-image {{
        max-height: 250px;
    }}
}}
"""

class MonoVisionModernInterface:
    """Modern Gradio interface for MonoVision V2 with dark theme and enhanced features"""
    
    def __init__(self):
        self.orchestrator: Optional[MonoVisionOrchestrator] = None
        self.chat_history = []
        self.current_tier = "fast"
        self.current_image_hash = None
        self.current_image_data = None
        self.last_technical_details = "No analysis performed yet"
        self.smart_interaction = SmartImageInteraction()  # Add smart interaction handler
        self.suggested_questions = []  # Store suggested questions
        self.current_objects = []  # Store current detected objects
        
        # Enhanced V3 features (if available)
        self.enhanced_chatbot = None
        self.production_dashboard = None
        self.mode_selector = None
        self.object_processor = None
        
        # Initialize enhanced features if available
        if ENHANCED_FEATURES_AVAILABLE:
            try:
                self.production_dashboard = ProductionDashboard()
                self.mode_selector = IntelligentModeSelector()
                print("🎯 Enhanced dashboard and mode selector initialized")
            except Exception as e:
                print(f"⚠️ Could not initialize enhanced features: {e}")
                self.production_dashboard = None
                self.mode_selector = None
        
    def get_image_hash(self, image_data: bytes) -> str:
        """Generate hash for image data"""
        return hashlib.md5(image_data).hexdigest()
    
    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string for display"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    
    def update_image_preview(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """Update image preview and info when new image is uploaded"""
        if image is None:
            return None, "No image uploaded"
        
        try:
            # Store current image data
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            self.current_image_data = img_byte_arr.getvalue()
            self.current_image_hash = self.get_image_hash(self.current_image_data)
            
            # Get image info
            width, height = image.size
            format_name = image.format or "PNG"
            size_kb = len(self.current_image_data) / 1024
            
            # Create image info text
            info_text = f"""**📋 Image Information:**
            
**📐 Dimensions:** {width} x {height} pixels  
**📄 Format:** {format_name}  
**💾 Size:** {size_kb:.1f} KB  
**🔑 Hash:** `{self.current_image_hash[:16]}...`  

**🎯 Ready for AI analysis!**"""
            
            return image, info_text
            
        except Exception as e:
            logger.error(f"❌ Error updating image preview: {e}")
            return None, f"❌ Error: {str(e)}"
        
    async def initialize(self):
        """Initialize the orchestrator"""
        try:
            logger.info("🚀 Initializing MonoVision V2 Modern Interface")
            
            # Initialize orchestrator synchronously to avoid event loop issues
            self.orchestrator = MonoVisionOrchestrator()
            
            # Try async initialization first
            try:
                await self.orchestrator.initialize()
            except Exception as init_error:
                logger.warning(f"⚠️ Async initialization failed, trying sync fallback: {init_error}")
                # Fallback: Initialize components manually if async fails
                try:
                    # Initialize cache manager synchronously
                    if hasattr(self.orchestrator, 'cache_manager') and self.orchestrator.cache_manager:
                        # Cache manager is already initialized during construction
                        pass
                    
                    # Initialize other components that might need manual setup
                    if hasattr(self.orchestrator, 'resource_manager'):
                        # Resource manager should be initialized
                        pass
                        
                    logger.info("✅ Fallback initialization successful")
                    
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback initialization also failed: {fallback_error}")
                    return False
            
            # Initialize enhanced V3 features if available
            if ENHANCED_FEATURES_AVAILABLE and self.orchestrator:
                try:
                    # Initialize enhanced chatbot system
                    self.enhanced_chatbot = EnhancedChatbotSystem(self.orchestrator)
                    
                    # Initialize object-specific processor
                    self.object_processor = ObjectSpecificQueryProcessor(self.orchestrator.vision_fusion)
                    
                    logger.info("🎯 Enhanced V3 chatbot features initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Could not initialize enhanced chatbot: {e}")
                    self.enhanced_chatbot = None
                    self.object_processor = None
            
            logger.info("✅ Modern Interface ready")
            return True
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def initialize_sync(self):
        """Synchronous initialization for when async fails"""
        try:
            logger.info("🚀 Initializing MonoVision V2 Modern Interface (Sync Mode)")
            
            # Initialize orchestrator
            self.orchestrator = MonoVisionOrchestrator()
            
            # Try to initialize components synchronously
            try:
                # Initialize cache manager (should be synchronous)
                if hasattr(self.orchestrator, 'cache_manager'):
                    # Cache manager initialization
                    pass
                
                # Initialize resource manager
                if hasattr(self.orchestrator, 'resource_manager'):
                    # Resource manager initialization
                    pass
                
                # For vision fusion, try synchronous initialization
                if hasattr(self.orchestrator, 'vision_fusion'):
                    try:
                        # Try to initialize vision components synchronously
                        # This might work if the models are cached
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self.orchestrator.vision_fusion.initialize())
                        loop.close()
                        logger.info("✅ Vision fusion initialized synchronously")
                    except Exception as vf_error:
                        logger.warning(f"⚠️ Vision fusion sync init failed: {vf_error}")
                
                logger.info("✅ Core components initialized (sync mode)")
                
            except Exception as comp_error:
                logger.warning(f"⚠️ Component initialization failed: {comp_error}")
                # Continue anyway - some components might still work
            
            # Initialize enhanced V3 features if available
            if ENHANCED_FEATURES_AVAILABLE and self.orchestrator:
                try:
                    # Initialize enhanced chatbot system
                    self.enhanced_chatbot = EnhancedChatbotSystem(self.orchestrator)
                    
                    # Initialize object-specific processor  
                    self.object_processor = ObjectSpecificQueryProcessor(self.orchestrator.vision_fusion)
                    
                    logger.info("🎯 Enhanced V3 chatbot features initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Could not initialize enhanced chatbot: {e}")
                    self.enhanced_chatbot = None
                    self.object_processor = None
            
            logger.info("✅ Modern Interface ready (sync mode)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Sync initialization failed: {e}")
            return False
    
    def get_cache_stats(self):
        """Get comprehensive cache statistics"""
        try:
            if not self.orchestrator or not self.orchestrator.cache_manager:
                return {
                    "ram_cache": {"results": {"entries": 0, "size_mb": 0}, "embeddings": {"entries": 0, "size_mb": 0}},
                    "database": {"results_count": 0, "embeddings_count": 0},
                    "performance": {"hit_rate_percent": 0}
                }
            
            # Get cache stats synchronously (the method is already sync)
            stats = self.orchestrator.cache_manager.get_stats()
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting cache stats: {e}")
            return {
                "ram_cache": {"results": {"entries": 0, "size_mb": 0}, "embeddings": {"entries": 0, "size_mb": 0}},
                "database": {"results_count": 0, "embeddings_count": 0},
                "performance": {"hit_rate_percent": 0}
            }
    
    def get_system_metrics(self):
        """Get current system performance metrics with enhanced cache info"""
        try:
            if not self.orchestrator:
                return "❌ System not initialized"
            
            # Get GPU memory info
            gpu_info = self.orchestrator.resource_manager.get_memory_info()
            
            # Get cache stats
            cache_stats = self.orchestrator.cache_manager.get_stats()
            
            # Get cache performance
            cache_perf = cache_stats.get('performance', {})
            cache_db = cache_stats.get('database', {})
            
            # Extract GPU info properly
            gpu_data = gpu_info.get('gpu', {})
            device_name = gpu_data.get('device_name', 'Unknown GPU')
            memory_allocated = gpu_data.get('memory_allocated_mb', 0)
            memory_total = gpu_data.get('memory_total_mb', 0)
            
            metrics = [
                f"🎯 **GPU**: {device_name}",
                f"📊 **VRAM**: {memory_allocated:.0f}MB / {memory_total:.0f}MB",
                f"⚡ **Mode**: {self.current_tier.title()}",
                f"🔄 **Cache Hit Rate**: {cache_perf.get('hit_rate_percent', 0):.1f}%",
                f"💾 **Cached Results**: {cache_db.get('results_count', 0)}",
                f"🔍 **Cached Visions**: {cache_db.get('results_count', 0) // 2}",  # Rough estimate
                f"📝 **Current Image**: {self.current_image_hash[:12] + '...' if self.current_image_hash else 'None'}"
            ]
            
            return "\n".join(metrics)
        except Exception as e:
            return f"❌ Error getting metrics: {e}"
    
    def get_enhanced_system_metrics(self):
        """Get enhanced system metrics with V3 dashboard features"""
        basic_metrics = self.get_system_metrics()
        
        if not ENHANCED_FEATURES_AVAILABLE or not self.production_dashboard:
            return basic_metrics
        
        try:
            # Get enhanced metrics from production dashboard
            enhanced_metrics = self.production_dashboard.get_system_metrics()
            
            # Format enhanced metrics
            enhanced_parts = []
            enhanced_parts.append("**🎯 Enhanced V3 Metrics:**")
            
            if 'performance' in enhanced_metrics:
                perf = enhanced_metrics['performance']
                enhanced_parts.append(f"📊 Avg Response Time: {perf.get('avg_response_time', 0):.2f}s")
                enhanced_parts.append(f"🎯 Success Rate: {perf.get('success_rate', 0):.1f}%")
            
            if 'resource_usage' in enhanced_metrics:
                resource = enhanced_metrics['resource_usage']
                enhanced_parts.append(f"🔋 CPU Usage: {resource.get('cpu_percent', 0):.1f}%")
                enhanced_parts.append(f"💾 Memory Usage: {resource.get('memory_percent', 0):.1f}%")
            
            if 'cache_analytics' in enhanced_metrics:
                cache = enhanced_metrics['cache_analytics']
                enhanced_parts.append(f"💾 Cache Efficiency: {cache.get('efficiency_score', 0):.1f}/10")
            
            return basic_metrics + "\n\n" + "\n".join(enhanced_parts)
            
        except Exception as e:
            logger.debug(f"Could not get enhanced metrics: {e}")
            return basic_metrics + "\n\n**🎯 Enhanced Metrics:** Loading..."
    
    def update_dashboard_stats(self, processing_time: float, mode: str, success: bool, cached: bool = False):
        """Update production dashboard with processing statistics"""
        if ENHANCED_FEATURES_AVAILABLE and self.production_dashboard:
            try:
                self.production_dashboard._update_session_stats(mode, processing_time, success)
                logger.debug(f"Dashboard updated: {processing_time:.2f}s, {mode}, success={success}, cached={cached}")
            except Exception as e:
                logger.warning(f"Failed to update dashboard stats: {e}")
    
    def get_dashboard_summary(self) -> str:
        """Get enhanced dashboard summary HTML for display"""
        if ENHANCED_FEATURES_AVAILABLE and self.production_dashboard:
            try:
                # Get comprehensive metrics from production dashboard
                metrics = self.production_dashboard.get_system_metrics()
                cache_stats = self.get_cache_stats()
                
                # Create detailed dashboard HTML
                dashboard_html = f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; margin: 10px 0; color: white;">
    <h3 style="margin-top: 0; text-align: center; color: white;">📊 Production Dashboard V3 Enhanced</h3>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0;">
        <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px;">
            <h4 style="margin-top: 0; color: #e0e0e0;">⚡ Performance Metrics</h4>
            <div style="font-size: 14px; line-height: 1.6;">
                • Avg Response: {metrics.get('performance', {}).get('avg_response_time', 0):.2f}s<br>
                • Success Rate: {metrics.get('performance', {}).get('success_rate', 100):.1f}%<br>
                • Total Requests: {metrics.get('performance', {}).get('total_requests', 0)}
            </div>
        </div>
        
        <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px;">
            <h4 style="margin-top: 0; color: #e0e0e0;">🖥️ System Resources</h4>
            <div style="font-size: 14px; line-height: 1.6;">
                • CPU: {metrics.get('resource_usage', {}).get('cpu_percent', 0):.1f}%<br>
                • Memory: {metrics.get('resource_usage', {}).get('memory_percent', 0):.1f}%<br>
                • GPU: {metrics.get('gpu_metrics', {}).get('gpu_utilization_percent', 0):.1f}%
            </div>
        </div>
    </div>
    
    <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px; margin-top: 10px;">
        <h4 style="margin-top: 0; color: #e0e0e0;">💾 Cache Analytics</h4>
        <div style="font-size: 14px; line-height: 1.6;">
            • Hit Rate: {cache_stats.get('performance', {}).get('hit_rate_percent', 0):.1f}%<br>
            • Cached Results: {cache_stats.get('database', {}).get('results_count', 0)}<br>
            • Cache Efficiency: {metrics.get('cache_analytics', {}).get('efficiency_score', 5.0):.1f}/10<br>
            • DB Size: {cache_stats.get('database', {}).get('db_size_mb', 0):.1f}MB
        </div>
    </div>
</div>
"""
                return dashboard_html
                
            except Exception as e:
                logger.warning(f"Failed to get enhanced dashboard summary: {e}")
                return self._get_fallback_dashboard_summary()
        else:
            return self._get_fallback_dashboard_summary()
    
    def _get_fallback_dashboard_summary(self) -> str:
        """Fallback dashboard summary when enhanced features are not available"""
        cache_stats = self.get_cache_stats()
        
        return f"""
<div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #007bff;">
    <h3 style="margin-top: 0; color: #495057;">📊 Basic Monitoring Dashboard</h3>
    
    <div style="background: white; padding: 10px; border-radius: 6px; margin: 10px 0;">
        <strong>Cache Statistics:</strong><br>
        • Hit Rate: {cache_stats.get('performance', {}).get('hit_rate_percent', 0):.1f}%<br>
        • Cached Results: {cache_stats.get('database', {}).get('results_count', 0)}<br>
        • RAM Cache Entries: {cache_stats.get('ram_cache', {}).get('results', {}).get('entries', 0)}
    </div>
    
    <div style="background: white; padding: 10px; border-radius: 6px; margin: 10px 0;">
        <strong>System Status:</strong><br>
        • Processing Mode: {self.current_tier.title()}<br>
        • Current Image: {self.current_image_hash[:12] + '...' if self.current_image_hash else 'None'}<br>
        • Session Active: ✅ Yes
    </div>
</div>
"""
    
    def get_suggested_questions(self) -> str:
        """Get current suggested questions as markdown"""
        if not self.suggested_questions:
            return "**💡 Suggested Questions:**\n\nUpload an image to get AI-generated questions!"
        
        questions_md = "**💡 Suggested Questions:**\n\n"
        for i, question in enumerate(self.suggested_questions[:6], 1):
            questions_md += f"**{i}.** {question}\n"
        
        questions_md += "\n*Click on detected objects in the image for specific questions!*"
        return questions_md
    
    def _get_enhanced_cache_status(self, response, processing_time: float) -> str:
        """Enhanced cache status detection with multiple indicators"""
        try:
            # Primary cache check from response object
            if hasattr(response, 'cached') and response.cached:
                return "✅ Yes (Response cached)"
            
            # Check for cache manager statistics and response characteristics
            cache_stats = self.get_cache_stats()
            db_results = cache_stats.get('database', {}).get('results_count', 0)
            ram_entries = cache_stats.get('ram_cache', {}).get('results', {}).get('entries', 0)
            hit_rate = cache_stats.get('performance', {}).get('hit_rate_percent', 0)
            
            # If we have cache data and reasonable processing time
            if db_results > 0 and processing_time < 10.0:
                # Very fast response with cache available = likely cached
                if processing_time < 2.0:
                    return f"✅ Likely (Fast: {processing_time:.2f}s, Cache: {db_results} entries)"
                elif processing_time < 5.0:
                    return f"🔄 Partial (Mixed: {processing_time:.2f}s, Hit rate: {hit_rate:.1f}%)"
                else:
                    return f"⚡ Some cached (Time: {processing_time:.2f}s, DB: {db_results})"
            
            # Check if this looks like a fresh model load (very slow first time)
            if processing_time > 25.0:
                return f"🔄 Fresh models loaded ({processing_time:.1f}s - initial run)"
            
            # Check for vision model cache indicators
            if hasattr(response, 'vision_results') and response.vision_results:
                vision_time = response.vision_results.get('vision_metrics', {}).get('total_time', processing_time)
                if vision_time < processing_time * 0.5:  # Vision was much faster than total
                    return f"🔍 Vision optimized (Models ready: {vision_time:.1f}s of {processing_time:.1f}s)"
            
            # Fast processing without obvious cache indicators = models already loaded
            if processing_time < 8.0:
                return f"⚡ Models ready (Efficient: {processing_time:.1f}s)"
            
            # Default to full processing
            return f"❌ No (Full processing: {processing_time:.1f}s)"
            
        except Exception as e:
            logger.debug(f"Error in enhanced cache status detection: {e}")
            return "❓ Unknown"
    
    def get_mode_recommendation(self, query: str, has_image: bool = False) -> str:
        """Get AI-powered mode recommendation"""
        if not ENHANCED_FEATURES_AVAILABLE or not self.mode_selector:
            # Fallback to simple logic
            if not query or len(query.strip()) < 10:
                return "**🎯 Mode Recommendation:** Balanced (General purpose)"
            elif len(query) > 100 or "detail" in query.lower() or "analyze" in query.lower():
                return "**🎯 Mode Recommendation:** Rich (Detailed analysis requested)"
            else:
                return "**🎯 Mode Recommendation:** Fast (Quick query detected)"
        
        try:
            # Use enhanced mode selector
            recommendation = self.mode_selector.recommend_mode(query, image_available=has_image)
            
            recommended_mode = recommendation['recommended_mode'].title()
            confidence = recommendation['confidence']
            reasoning = recommendation['reasoning'][:100] + "..." if len(recommendation['reasoning']) > 100 else recommendation['reasoning']
            
            return f"""**🧠 AI Mode Recommendation:** {recommended_mode}
            
**Confidence:** {confidence:.0%}
**Reasoning:** {reasoning}

*AI automatically selected this mode based on your query complexity and requirements.*"""
            
        except Exception as e:
            logger.debug(f"Could not get enhanced mode recommendation: {e}")
            return "**🎯 Mode Recommendation:** Balanced (Default fallback)"
    
    def create_interactive_overlay(self, image: Image.Image, objects: List[Dict[str, Any]]) -> str:
        """Create interactive HTML overlay for detected objects"""
        try:
            overlay_gen = ObjectOverlayGenerator()
            overlay_data = overlay_gen.create_overlay_data(image, objects)
            
            if 'interactive_html' in overlay_data:
                return overlay_data['interactive_html']
            else:
                # Fallback to simple image display
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG", quality=90)
                image_b64 = base64.b64encode(buffered.getvalue()).decode()
                return f'<img src="data:image/jpeg;base64,{image_b64}" style="max-width: 100%; height: auto; border-radius: 8px;">'
                
        except Exception as e:
            logger.error(f"❌ Error creating interactive overlay: {e}")
            return f"<div style='color: red;'>Error creating overlay: {e}</div>"
    
    def get_technical_details(self):
        """Get the latest technical analysis details"""
        return getattr(self, 'last_technical_details', "No analysis performed yet")
    
    def process_image_and_text_streaming(
        self, 
        image_file,
        text_query: str,
        processing_mode: str,
        include_objects: bool,
        send_image_to_gemini: bool,
        chat_history: List
    ):
        """Process image with text query using selected mode with streaming output"""
        
        if not self.orchestrator:
            error_msg = "❌ System not initialized"
            chat_history.append({"role": "user", "content": text_query or "[Image uploaded]"})
            chat_history.append({"role": "assistant", "content": error_msg})
            yield "", "", chat_history, "System not ready"
            return
        
        try:
            start_time = time.time()
            self.current_tier = processing_mode.lower()
            
            # Initialize variables that might be used in different code paths
            objects = []
            
            # Add user message to chat immediately
            user_message = text_query or "📷 *Uploaded an image for analysis*"
            chat_history.append({"role": "user", "content": user_message})
            
            # Yield initial state - show user message and empty AI response
            yield "", "", chat_history, self.get_system_metrics()
            
            # Handle image upload
            if image_file is not None:
                # Show processing status
                chat_history[-1]["content"] = "🔄 Processing image..."
                yield "", "", chat_history, self.get_system_metrics()
                
                # Convert image to bytes and get hash
                if hasattr(image_file, 'save'):
                    # PIL Image
                    img_byte_arr = io.BytesIO()
                    image_file.save(img_byte_arr, format='PNG')
                    image_bytes = img_byte_arr.getvalue()
                    display_image = image_file
                else:
                    # File path
                    with open(image_file, 'rb') as f:
                        image_bytes = f.read()
                    display_image = Image.open(image_file)
                
                # Generate image hash for caching
                image_hash = self.get_image_hash(image_bytes)
                self.current_image_hash = image_hash
                self.current_image_data = image_bytes
                
                # Update processing status
                chat_history[-1]["content"] = "🔍 Analyzing image..."
                yield "", "", chat_history, self.get_system_metrics()
                
                # Check if we have cached vision results
                cached_vision = None
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    cached_vision = loop.run_until_complete(
                        self.orchestrator.cache_manager.get_cached_vision_result(image_hash)
                    )
                    loop.close()
                except:
                    pass
                
                if cached_vision:
                    chat_history[-1]["content"] = "⚡ Found cached analysis..."
                    yield "", "", chat_history, self.get_system_metrics()

                # Create processing request with stable session ID for context memory
                # Use a simple stable session ID for this browser session
                stable_session_id = "gradio_main_session"
                
                request = ProcessingRequest(
                    image_data=image_bytes,
                    query=text_query or "Describe this image in detail",
                    mode=ProcessingMode(processing_mode.lower()),
                    session_id=stable_session_id,
                    request_id=f"req_{int(time.time())}",
                    include_objects=include_objects,
                    send_image_to_gemini=send_image_to_gemini
                )                # Update processing status
                chat_history[-1]["content"] = f"🚀 Running {processing_mode} analysis..."
                yield "", "", chat_history, self.get_system_metrics()
                
                # Process asynchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(self.orchestrator.process_request(request))
                loop.close()
                
                # Update with AI generation status
                chat_history[-1]["content"] = "🤖 Generating response..."
                yield "", "", chat_history, self.get_system_metrics()
                
                # Extract objects from response for interactions
                objects = response.vision_results.get('objects', []) if response.vision_results else []
                
                # Apply enhanced object detection filtering
                if objects and include_objects:
                    try:
                        # Get image dimensions for size filtering
                        image_size = (display_image.size[0], display_image.size[1])
                        objects = enhanced_object_detection(objects, image_size)
                        logger.info(f"Enhanced object detection: {len(objects)} objects after filtering")
                    except Exception as e:
                        logger.warning(f"Enhanced object detection failed, using original: {e}")
                
                # Store current objects for interactions
                self.current_objects = objects
                
                # Perform enhanced image analysis (skip if sending to Gemini API)
                enhanced_analysis = {}
                if not send_image_to_gemini:
                    try:
                        enhanced_analysis = enhanced_image_analysis(display_image, response.vision_results)
                        logger.info("Enhanced image analysis completed")
                    except Exception as e:
                        logger.warning(f"Enhanced image analysis failed: {e}")
                else:
                    logger.info("⏭️ Skipping enhanced image analysis (sending to Gemini API)")
                
                # Process with smart interaction system
                interaction_result = self.smart_interaction.process_image_upload(display_image, objects)
                self.suggested_questions = interaction_result.get('suggested_questions', [])
                
                # Clean the AI response first - remove any technical artifacts
                clean_ai_response = response.language_response.strip()
                
                # Remove any technical logging that might have leaked in
                technical_patterns = [
                    "BLIP:", "CLIP:", "YOLO:", "Vision:", "Processing:",
                    "Generated in", "tokens", "Time:", "Model:", 
                    "Loading", "Initialized", "Cache:", "GPU:"
                ]
                
                for pattern in technical_patterns:
                    # Remove lines containing technical information
                    lines = clean_ai_response.split('\n')
                    clean_lines = []
                    for line in lines:
                        if not any(pattern.lower() in line.lower() for pattern in technical_patterns):
                            clean_lines.append(line)
                    clean_ai_response = '\n'.join(clean_lines)
                
                # Stream the response word by word
                if clean_ai_response and len(clean_ai_response.strip()) > 10:
                    words = clean_ai_response.split()
                    current_response = ""

                    # Set up the initial chat structure
                    chat_history.append({"role": "user", "content": user_message})
                    chat_history.append({"role": "assistant", "content": ""})

                    for i, word in enumerate(words):
                        current_response += word + " "
                        # Only update the assistant message content, don't recreate the structure
                        chat_history[-1]["content"] = current_response.rstrip()

                        # Add some delay for natural typing effect
                        if i % 3 == 0:  # Update every 3 words
                            yield "", "", chat_history, self.get_system_metrics()
                            time.sleep(0.05)  # Small delay for typing effect
                else:
                    # Fallback response
                    fallback_response = "I've analyzed your image! Feel free to ask me anything specific about what you'd like to know."
                    words = fallback_response.split()
                    current_response = ""

                    # Set up the initial chat structure
                    chat_history.append({"role": "user", "content": user_message})
                    chat_history.append({"role": "assistant", "content": ""})

                    for i, word in enumerate(words):
                        current_response += word + " "
                        # Only update the assistant message content
                        chat_history[-1]["content"] = current_response.rstrip()

                        if i % 2 == 0:
                            yield "", "", chat_history, self.get_system_metrics()
                            time.sleep(0.05)
                
                # Final response with all enhancements
                processing_time = time.time() - start_time
                
                # Add context-aware enhancements to the response
                response_enhancements = []
                
                # Add object detection insights if available
                if response.vision_results.get('objects') and include_objects:
                    if len(objects) > 0:
                        main_objects = [obj.get('name', obj.get('class', 'object')) for obj in objects[:3]]
                        response_enhancements.append(f"I can detect {len(objects)} object{'s' if len(objects) > 1 else ''} in the image: {', '.join(main_objects)}.")
                
                # Add mood/style insights from CLIP if available
                if response.vision_results.get('clip_keywords'):
                    keywords = response.vision_results['clip_keywords'][:3]
                    style_words = [kw for kw in keywords if kw.lower() in ['modern', 'vintage', 'artistic', 'natural', 'urban', 'peaceful', 'bright', 'dark']]
                    if style_words:
                        response_enhancements.append(f"The overall style feels {', '.join(style_words)}.")
                
                # Build the final chatbot response
                final_response = clean_ai_response
                if response_enhancements:
                    final_response += "\n\n" + " ".join(response_enhancements)
                
                # Show final response
                chat_history[-1]["content"] = final_response
                
                # Create processing summary for technical details with enhanced cache detection
                cache_status = self._get_enhanced_cache_status(response, processing_time)
                
                processing_summary = f"""**🔬 Processing Summary:**

**⏱️ Performance:**
• Total time: {processing_time:.2f}s
• Mode: {processing_mode.title()}
• Cached: {cache_status}

**📊 Vision Results:**
• Caption quality: {'High' if len(response.vision_results.get('caption', '')) > 50 else 'Standard'}
• Objects detected: {len(objects)}
• CLIP keywords: {len(response.vision_results.get('clip_keywords', []))}

**🎯 Response Stats:**
• Response length: {len(final_response)} characters
• Processing tier: {self.current_tier}

**💾 Cache Status:**
• Hit Rate: {self.get_cache_stats().get('performance', {}).get('hit_rate_percent', 0):.1f}%
• DB Results: {self.get_cache_stats().get('database', {}).get('results_count', 0)}
"""
                
                # Store technical details
                self.last_technical_details = processing_summary
                
                # Final yield with complete state - no image file to avoid permission errors
                yield "", "", chat_history, self.get_system_metrics()
                
            else:
                # Enhanced text-only processing with proper language model
                chat_history[-1] = {"role": "user", "content": user_message}
                chat_history.append({"role": "assistant", "content": "🤖 Processing your question..."})
                yield "", "", chat_history, self.get_system_metrics()
                
                # Create proper text-only request
                request = ProcessingRequest(
                    image_data=b"",  # Empty image data for text-only
                    query=text_query or "Hello! How can I help you today?",
                    mode=ProcessingMode(processing_mode.lower()),
                    session_id="gradio_main_session", 
                    request_id=f"text_req_{int(time.time())}",
                    include_objects=False
                )
                
                try:
                    # Process with orchestrator - it will handle text-only mode
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(self.orchestrator.process_request(request))
                    loop.close()
                    
                    # Get the language response
                    ai_response = response.language_response.strip()
                    
                    # If response is still weak, enhance it
                    if len(ai_response) < 20 or any(weak in ai_response.lower() for weak in ["unclear", "unknown", "can't", "unable"]):
                        # Provide a more helpful fallback response
                        ai_response = f"""I'm here to help! I can assist you with:

🖼️ **Image Analysis**: Upload an image and I'll describe it, detect objects, and answer questions about it
💬 **General Chat**: Ask me about various topics and I'll do my best to help
🔍 **Technical Details**: I can explain how image processing works
⚙️ **Processing Modes**: Choose Fast, Balanced, or Rich for different levels of analysis

What would you like to explore today?"""
                    
                    # Stream the response word by word for better UX
                    words = ai_response.split()
                    current_response = ""

                    # Set up the initial chat structure
                    chat_history.append({"role": "user", "content": user_message})
                    chat_history.append({"role": "assistant", "content": ""})

                    for i, word in enumerate(words):
                        current_response += word + " "
                        # Only update the assistant message content
                        chat_history[-1]["content"] = current_response.rstrip()
                        yield "", "", chat_history, self.get_system_metrics()
                        time.sleep(0.03)  # Faster streaming for text
                    
                except Exception as e:
                    # Enhanced error handling with helpful message
                    error_response = f"""I encountered an issue processing your request: {str(e)}

However, I'm still here to help! You can:
• Try rephrasing your question
• Upload an image for visual analysis  
• Ask me about general topics
• Check the system status in the sidebar

What would you like to try?"""
                    
                    # Stream error response
                    words = error_response.split()
                    current_response = ""

                    # Set up the initial chat structure
                    chat_history.append({"role": "user", "content": user_message})
                    chat_history.append({"role": "assistant", "content": ""})

                    for i, word in enumerate(words):
                        current_response += word + " "
                        # Only update the assistant message content
                        chat_history[-1]["content"] = current_response.rstrip()
                        yield "", "", chat_history, self.get_system_metrics()
                        time.sleep(0.03)
                
                # Update metrics
                yield "", None, chat_history, self.get_system_metrics()
                
        except Exception as e:
            error_msg = f"❌ **Processing Error**: {str(e)}"
            chat_history.append({"role": "user", "content": text_query or "[Error]"})
            chat_history.append({"role": "assistant", "content": error_msg})
            logger.error(f"Processing error: {e}")
            yield "", None, chat_history, "❌ Error occurred"

    def process_image_and_text(
        self, 
        image_file,
        text_query: str,
        processing_mode: str,
        include_objects: bool,
        send_image_to_gemini: bool,
        chat_history: List
    ) -> Tuple[str, str, List, str]:
        """Process image with text query using selected mode with enhanced caching and preview (non-streaming)"""
        
        if not self.orchestrator:
            error_msg = "❌ System not initialized"
            chat_history.append({"role": "user", "content": text_query or "[Image uploaded]"})
            chat_history.append({"role": "assistant", "content": error_msg})
            return "", "", chat_history, "System not ready"
        
        try:
            start_time = time.time()
            self.current_tier = processing_mode.lower()
            
            # Initialize variables that might be used in different code paths
            objects = []
            
            # Handle image upload
            if image_file is not None:
                # Convert image to bytes and get hash
                if hasattr(image_file, 'save'):
                    # PIL Image
                    img_byte_arr = io.BytesIO()
                    image_file.save(img_byte_arr, format='PNG')
                    image_bytes = img_byte_arr.getvalue()
                    display_image = image_file
                else:
                    # File path
                    with open(image_file, 'rb') as f:
                        image_bytes = f.read()
                    display_image = Image.open(image_file)
                
                # Generate image hash for caching
                image_hash = self.get_image_hash(image_bytes)
                self.current_image_hash = image_hash
                self.current_image_data = image_bytes
                
                # Check if we have cached vision results
                cached_vision = None
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    cached_vision = loop.run_until_complete(
                        self.orchestrator.cache_manager.get_cached_vision_result(image_hash)
                    )
                    loop.close()
                except:
                    pass
                
                # Create processing request with stable session ID
                request = ProcessingRequest(
                    image_data=image_bytes,
                    query=text_query or "Describe this image in detail",
                    mode=ProcessingMode(processing_mode.lower()),
                    session_id="gradio_main_session",
                    request_id=f"req_{int(time.time())}",
                    include_objects=include_objects,
                    send_image_to_gemini=send_image_to_gemini
                )
                
                # Process asynchronously
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(self.orchestrator.process_request(request))
                loop.close()
                
                # Format response with image preview
                processing_time = time.time() - start_time
                
                # Create simple text-based image info (no HTML)
                image_info = f"📷 **Image Upload** - Hash: `{image_hash[:12]}...` | Size: {len(image_bytes):,} bytes"
                
                # Extract objects from response for interactions
                objects = response.vision_results.get('objects', []) if response.vision_results else []
                
                # Apply enhanced object detection filtering
                if objects and include_objects:
                    try:
                        # Get image dimensions for size filtering
                        image_size = (display_image.size[0], display_image.size[1])
                        objects = enhanced_object_detection(objects, image_size)
                        logger.info(f"Enhanced object detection: {len(objects)} objects after filtering")
                    except Exception as e:
                        logger.warning(f"Enhanced object detection failed, using original: {e}")
                
                # Store current objects for interactions
                self.current_objects = objects
                
                # Perform enhanced image analysis (skip if sending to Gemini API)
                enhanced_analysis = {}
                if not send_image_to_gemini:
                    try:
                        enhanced_analysis = enhanced_image_analysis(display_image, response.vision_results)
                        logger.info("Enhanced image analysis completed")
                    except Exception as e:
                        logger.warning(f"Enhanced image analysis failed: {e}")
                else:
                    logger.info("⏭️ Skipping enhanced image analysis (sending to Gemini API)")
                
                # Process with smart interaction system
                interaction_result = self.smart_interaction.process_image_upload(display_image, objects)
                self.suggested_questions = interaction_result.get('suggested_questions', [])
                
                # === ENHANCED V3 PROCESSING PATH ===
                if ENHANCED_FEATURES_AVAILABLE and self.enhanced_chatbot:
                    try:
                        # Use enhanced chatbot system for better responses (run in event loop)
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        enhanced_response = loop.run_until_complete(
                            self.enhanced_chatbot.process_intelligent_query(
                                query=text_query or "Describe this image in detail",
                                image=display_image,
                                session_id="gradio_main_session",
                                preferred_mode=processing_mode.lower()
                            )
                        )
                        loop.close()
                        
                        # Use enhanced response if available
                        if enhanced_response and enhanced_response.text:
                            clean_ai_response = enhanced_response.text.strip()
                            
                            # Add enhanced suggestions if available
                            if enhanced_response.suggestions:
                                self.suggested_questions = enhanced_response.suggestions[:6]
                            
                            # Merge vision insights
                            if enhanced_response.vision_insights:
                                response.vision_results.update(enhanced_response.vision_insights)
                                
                        else:
                            # Fallback to standard processing
                            clean_ai_response = response.language_response.strip()
                            
                    except Exception as e:
                        logger.debug(f"Enhanced chatbot failed, using standard: {e}")
                        # Fallback to standard processing
                        clean_ai_response = response.language_response.strip()
                else:
                    # Standard V2 processing
                    clean_ai_response = response.language_response.strip()
                
                # === CLEAN RESPONSE PROCESSING ===
                
                # Remove any technical logging that might have leaked in
                technical_patterns = [
                    "BLIP:", "CLIP:", "YOLO:", "Vision:", "Processing:",
                    "Generated in", "tokens", "Time:", "Model:", 
                    "Loading", "Initialized", "Cache:", "GPU:"
                ]
                
                for pattern in technical_patterns:
                    # Remove lines containing technical information
                    lines = clean_ai_response.split('\n')
                    clean_lines = []
                    for line in lines:
                        if not any(pattern.lower() in line.lower() for pattern in technical_patterns):
                            clean_lines.append(line)
                    clean_ai_response = '\n'.join(clean_lines)
                
                # Enhance the AI response to be more conversational
                if not clean_ai_response or len(clean_ai_response.strip()) < 10:
                    # Create a natural response based on vision results
                    if 'caption' in response.vision_results:
                        clean_ai_response = f"I can see {response.vision_results['caption'].lower()}. "
                        if text_query and text_query.strip():
                            clean_ai_response += f"You asked about '{text_query}' - let me know if you'd like me to focus on any particular aspect of what's shown!"
                    else:
                        clean_ai_response = "I've analyzed your image! Feel free to ask me anything specific about what you'd like to know."
                
                # Add context-aware enhancements to the response
                response_enhancements = []
                
                # Add object detection insights if available
                if response.vision_results.get('objects') and include_objects:
                    if len(objects) > 0:
                        main_objects = [obj.get('name', obj.get('class', 'object')) for obj in objects[:3]]
                        response_enhancements.append(f"I can detect {len(objects)} object{'s' if len(objects) > 1 else ''} in the image: {', '.join(main_objects)}.")
                
                # Add mood/style insights from CLIP if available
                if response.vision_results.get('clip_keywords'):
                    keywords = response.vision_results['clip_keywords'][:3]
                    style_words = [kw for kw in keywords if kw.lower() in ['modern', 'vintage', 'artistic', 'natural', 'urban', 'peaceful', 'bright', 'dark']]
                    if style_words:
                        response_enhancements.append(f"The overall style feels {', '.join(style_words)}.")
                
                # Build the final chatbot response
                chatbot_response_parts = []
                
                # Main AI response
                chatbot_response_parts.append(clean_ai_response)
                
                # Add conversational enhancements
                if response_enhancements:
                    chatbot_response_parts.append("")  # Add spacing
                    chatbot_response_parts.extend(response_enhancements)
                
                # Add a natural closing
                if text_query and text_query.strip():
                    chatbot_response_parts.append(f"\nIs there anything specific about the image you'd like me to elaborate on?")
                else:
                    chatbot_response_parts.append(f"\nFeel free to ask me anything about what you see!")
                
                # Final formatted response for chat
                formatted_response = "\n".join(chatbot_response_parts)
                
                # === CREATE SEPARATE TECHNICAL DETAILS PANEL ===
                
                # Create technical details for separate display
                tech_details_parts = []
                
                # Image information
                tech_details_parts.append(f"**📷 Image Analysis**")
                tech_details_parts.append(f"• Hash: `{image_hash[:16]}...`")
                tech_details_parts.append(f"• Size: {len(image_bytes):,} bytes")
                tech_details_parts.append(f"• Dimensions: {display_image.size[0]} × {display_image.size[1]} pixels")
                
                # Cache status with enhanced detection
                cache_status = "❌ No" 
                cache_details = ""
                
                # Primary cache check from response
                if hasattr(response, 'cached') and response.cached:
                    cache_status = "✅ Yes (Full Result)"
                    cache_details = "Complete response from cache"
                elif cached_vision:
                    cache_status = "🔄 Partial (Vision Only)"
                    cache_age_minutes = int((time.time() - cached_vision.get('timestamp', 0))/60)
                    cache_details = f"Vision models cached {cache_age_minutes}m ago"
                else:
                    # Secondary detection: fast processing time might indicate caching
                    if processing_time < 5.0 and len(response.language_response) > 20:
                        cache_status = "🔄 Likely (Fast Response)"
                        cache_details = f"Response time {processing_time:.1f}s suggests caching"
                    else:
                        cache_details = "Fresh analysis performed"
                
                tech_details_parts.append(f"• Cache: {cache_status}")
                if cache_details:
                    tech_details_parts.append(f"  └─ {cache_details}")
                
                # Computer Vision Analysis
                tech_details_parts.append(f"\n**🔍 Vision Models**")
                
                # Get fusion results if available
                fusion_result = response.vision_results.get('fusion_result', {})
                
                if fusion_result and 'enhanced_description' in fusion_result:
                    tech_details_parts.append(f"• **BLIP+CLIP Fusion:** {fusion_result['enhanced_description']}")
                    if fusion_result.get('fusion_quality', 0) > 0.7:
                        tech_details_parts.append(f"• Quality Score: {fusion_result['fusion_quality']:.2f}/1.0 ✨")
                elif 'caption' in response.vision_results:
                    tech_details_parts.append(f"• **BLIP Caption:** {response.vision_results['caption']}")
                
                # Add CLIP keywords if available
                if response.vision_results.get('clip_keywords'):
                    keywords = response.vision_results['clip_keywords'][:5]
                    tech_details_parts.append(f"• **CLIP Keywords:** {', '.join(keywords)}")
                
                # Object Detection
                if response.vision_results.get('objects') and include_objects:
                    objects = response.vision_results['objects']
                    tech_details_parts.append(f"• **YOLO Objects:** {len(objects)} detected")
                    for obj in objects[:3]:  # Show top 3
                        obj_name = obj.get('name', obj.get('class', 'unknown'))
                        confidence = obj.get('confidence', 0)
                        tech_details_parts.append(f"  - {obj_name} ({confidence:.2f})")
                    
                    # Add object overlay to technical details
                    try:
                        from monovision_v2.ui.enhanced_ui_components import ObjectOverlayGenerator
                        overlay_gen = ObjectOverlayGenerator()
                        overlay_data = overlay_gen.create_overlay_data(display_image, objects)
                        
                        if 'overlay_markdown' in overlay_data:
                            tech_details_parts.append(f"\n**📦 Object Detection Details:**")
                            tech_details_parts.append(overlay_data['overlay_markdown'])
                    except Exception as e:
                        logger.warning(f"Could not create object overlay: {e}")
                
                # Enhanced Analysis Results
                if enhanced_analysis:
                    tech_details_parts.append(f"\n**🎨 Enhanced Analysis**")
                    
                    # Color Analysis
                    if 'color_analysis' in enhanced_analysis and enhanced_analysis['color_analysis']:
                        color_info = enhanced_analysis['color_analysis']
                        if 'dominant_colors' in color_info:
                            tech_details_parts.append(f"• **Colors:** {', '.join(color_info['dominant_colors'][:3])}")
                        if 'temperature' in color_info:
                            tech_details_parts.append(f"• **Temperature:** {color_info['temperature']}")
                        if 'saturation' in color_info:
                            tech_details_parts.append(f"• **Saturation:** {color_info['saturation']}")
                    
                    # Composition Analysis
                    if 'composition' in enhanced_analysis and enhanced_analysis['composition']:
                        comp_info = enhanced_analysis['composition']
                        if 'orientation' in comp_info:
                            tech_details_parts.append(f"• **Orientation:** {comp_info['orientation']}")
                        if 'balance' in comp_info:
                            tech_details_parts.append(f"• **Balance:** {comp_info['balance']}")
                        if 'rule_of_thirds_alignment' in comp_info:
                            tech_details_parts.append(f"• **Rule of Thirds:** {comp_info['rule_of_thirds_alignment']} points")
                    
                    # Style and Mood
                    if 'style_mood' in enhanced_analysis and enhanced_analysis['style_mood']:
                        style_info = enhanced_analysis['style_mood']
                        if 'detected_styles' in style_info and style_info['detected_styles']:
                            tech_details_parts.append(f"• **Style:** {', '.join(style_info['detected_styles'][:2])}")
                        if 'detected_moods' in style_info and style_info['detected_moods']:
                            tech_details_parts.append(f"• **Mood:** {', '.join(style_info['detected_moods'][:2])}")
                    
                    # Scene Classification
                    if 'scene_type' in enhanced_analysis and enhanced_analysis['scene_type']:
                        scene_info = enhanced_analysis['scene_type']
                        if 'primary_scene' in scene_info:
                            tech_details_parts.append(f"• **Scene:** {scene_info['primary_scene']}")
                    
                    # Lighting Analysis
                    if 'lighting' in enhanced_analysis and enhanced_analysis['lighting']:
                        light_info = enhanced_analysis['lighting']
                        if 'overall_brightness' in light_info:
                            tech_details_parts.append(f"• **Lighting:** {light_info['overall_brightness']}")
                        if 'contrast_level' in light_info:
                            tech_details_parts.append(f"• **Contrast:** {light_info['contrast_level']}")
                    
                    # Texture Analysis
                    if 'texture' in enhanced_analysis and enhanced_analysis['texture']:
                        texture_info = enhanced_analysis['texture']
                        if 'texture_type' in texture_info:
                            tech_details_parts.append(f"• **Texture:** {texture_info['texture_type']}")
                    
                    # Emotional Impact
                    if 'emotional_impact' in enhanced_analysis and enhanced_analysis['emotional_impact']:
                        emotion_info = enhanced_analysis['emotional_impact']
                        if 'dominant_emotion' in emotion_info:
                            tech_details_parts.append(f"• **Emotion:** {emotion_info['dominant_emotion']}")
                
                # Performance Metrics
                tech_details_parts.append(f"\n**⚡ Performance**")
                vision_metrics = response.vision_results.get('vision_metrics', {})
                if vision_metrics:
                    if vision_metrics.get('blip_time', 0) > 0:
                        tech_details_parts.append(f"• BLIP: {vision_metrics.get('blip_time', 0):.2f}s")
                    if vision_metrics.get('clip_time', 0) > 0:
                        tech_details_parts.append(f"• CLIP: {vision_metrics.get('clip_time', 0):.2f}s") 
                    if vision_metrics.get('yolo_time', 0) > 0:
                        tech_details_parts.append(f"• YOLO: {vision_metrics.get('yolo_time', 0):.2f}s")
                tech_details_parts.append(f"• Total: {processing_time:.2f}s")
                tech_details_parts.append(f"• Tokens: {response.token_count}")
                tech_details_parts.append(f"• Mode: {response.mode.value.title()}")
                
                # Cache statistics
                try:
                    cache_stats = self.get_cache_stats()
                    if cache_stats:
                        hit_rate = cache_stats.get('performance', {}).get('hit_rate_percent', 0)
                        tech_details_parts.append(f"\n**💾 Cache Stats**")
                        tech_details_parts.append(f"• Hit Rate: {hit_rate:.1f}%")
                        tech_details_parts.append(f"• Results: {cache_stats.get('database', {}).get('results_count', 0)}")
                        total_size = sum([
                            cache_stats.get('ram_cache', {}).get('results', {}).get('size_mb', 0),
                            cache_stats.get('ram_cache', {}).get('embeddings', {}).get('size_mb', 0),
                            cache_stats.get('ram_cache', {}).get('images', {}).get('size_mb', 0)
                        ])
                        tech_details_parts.append(f"• Size: {total_size:.1f}MB")
                except Exception as e:
                    logger.debug(f"Could not get cache stats: {e}")
                
                technical_details = "\n".join(tech_details_parts)
                
                # Add user message and AI response to chat history in proper format for Gradio 5.x
                user_message = text_query or "📷 *Uploaded an image for analysis*"

                # Add small image preview to chat
                try:
                    if hasattr(image_file, 'resize'):
                        # Create a small thumbnail for chat preview
                        thumbnail = image_file.copy()
                        thumbnail.thumbnail((150, 150), Image.Resampling.LANCZOS)

                        # Convert to base64 for inline display
                        buffered = io.BytesIO()
                        thumbnail.save(buffered, format="JPEG", quality=80)
                        img_data = base64.b64encode(buffered.getvalue()).decode()

                        user_message += f'\n\n<img src="data:image/jpeg;base64,{img_data}" style="max-width: 150px; border-radius: 8px; border: 1px solid #444; margin-top: 8px;">'
                except Exception as e:
                    logger.debug(f"Could not add image preview to chat: {e}")

                # Use proper message format for Gradio 5.x type='messages'
                chat_history.append({"role": "user", "content": user_message})
                chat_history.append({"role": "assistant", "content": formatted_response})
                
                # Store technical details for separate panel display
                self.last_technical_details = technical_details
                
                # Update production dashboard stats
                self.update_dashboard_stats(
                    processing_time=processing_time,
                    mode=processing_mode,
                    success=True,
                    cached=getattr(response, 'cached', False)
                )
                
                # Cache the vision results for future use
                if not response.cached and 'caption' in response.vision_results:
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            self.orchestrator.cache_manager.cache_vision_result(
                                image_hash,
                                response.vision_results.get('caption', ''),
                                response.vision_results.get('keywords', []),
                                response.vision_results.get('objects', [])
                            )
                        )
                        loop.close()
                    except Exception as e:
                        logger.warning(f"⚠️ Could not cache vision result: {e}")
                
                # Return updated chat and clear inputs
                return "", None, chat_history, self.get_system_metrics()
                
            else:
                # Enhanced text-only processing with proper language model
                processing_query = text_query or "Hello! How can I help you today?"
                
                request = ProcessingRequest(
                    image_data=b"",  # Empty image data for text-only mode
                    query=processing_query,
                    mode=ProcessingMode(processing_mode.lower()),
                    session_id="gradio_main_session",
                    request_id=f"text_req_{int(time.time())}",
                    include_objects=False
                )
                
                try:
                    # Process asynchronously with orchestrator
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    response = loop.run_until_complete(self.orchestrator.process_request(request))
                    loop.close()
                    
                    # Format text-only response with enhanced styling
                    processing_time = time.time() - start_time
                    
                    # Clean and format the AI response properly
                    clean_response = response.language_response.replace('\\n', '\n').strip()
                    
                    # If response is weak, provide enhanced fallback
                    if len(clean_response) < 20 or any(weak in clean_response.lower() for weak in ["unclear", "unknown", "can't", "unable", "sorry"]):
                        clean_response = f"""I'm here to help you! Here's what I can do:

🖼️ **Image Analysis**: Upload any image and I'll:
   • Describe what I see in detail
   • Detect and identify objects
   • Answer specific questions about the image
   • Analyze composition, colors, and style

💬 **Conversation**: I can discuss:
   • Technology and AI topics
   • General knowledge questions
   • Help with various tasks
   • Explain how image processing works

⚙️ **Processing Modes**:
   • **Fast**: Quick responses (BLIP + Flan-T5)
   • **Balanced**: Better analysis (BLIP + CLIP + Phi-2)
   • **Rich**: Detailed insights (BLIP + CLIP + Mistral-7B)

What would you like to explore today? Feel free to ask me anything or upload an image for analysis!"""
                    
                    # Create natural chatbot response for text-only
                    chatbot_response_parts = []
                    
                    # Context indicator if we have a previous image
                    if self.current_image_hash:
                        chatbot_response_parts.append(f"💭 *I still remember the image you uploaded earlier (Hash: {self.current_image_hash[:12]}...)*\n")
                    
                    # Main AI response
                    chatbot_response_parts.append(clean_response)
                    
                    # Add conversational closing
                    if not any(question in clean_response for question in ["?", "What", "How", "explore"]):
                        chatbot_response_parts.append(f"\nIs there anything specific you'd like to know more about?")
                    
                    formatted_response = "\n".join(chatbot_response_parts)
                    
                except Exception as e:
                    # Enhanced error handling with helpful message
                    logger.error(f"Text processing error: {e}")
                    formatted_response = f"""I encountered an issue: {str(e)}

But I'm still here to help! You can:
• Try rephrasing your question
• Upload an image for visual analysis
• Ask me about general topics
• Check the system status in the sidebar

What would you like to try next?"""
                
                # Create technical details for text processing
                tech_details_parts = []
                tech_details_parts.append("**💬 Text Processing**")
                if self.current_image_hash:
                    tech_details_parts.append(f"• Context: Image `{self.current_image_hash[:12]}...`")
                tech_details_parts.append(f"• Processing: {processing_time:.2f}s")
                tech_details_parts.append(f"• Tokens: {response.token_count}")
                tech_details_parts.append(f"• Mode: {processing_mode.title()}")
                if response.cached:
                    tech_details_parts.append("• Source: Cached")
                
                self.last_technical_details = "\n".join(tech_details_parts)
                
                # Update production dashboard stats for text processing
                self.update_dashboard_stats(
                    processing_time=processing_time,
                    mode=processing_mode,
                    success=True,
                    cached=getattr(response, 'cached', False)
                )
                
                # Add to chat history with proper message format for Gradio 5.x
                chat_history.append({"role": "user", "content": f"💬 {text_query}"})
                chat_history.append({"role": "assistant", "content": formatted_response})
                
                # Update metrics
                metrics = self.get_system_metrics()
                
                return "", None, chat_history, metrics
                
        except Exception as e:
            error_msg = f"❌ **Processing Error**: {str(e)}"
            # Use proper message format for Gradio 5.x
            chat_history.append({"role": "user", "content": text_query or "[Error]"})
            chat_history.append({"role": "assistant", "content": error_msg})
            logger.error(f"Processing error: {e}")
            return "", None, chat_history, "❌ Error occurred"
    
    def clear_chat(self):
        """Clear chat history and reset context"""
        self.current_image_hash = None
        self.current_image_data = None
        return [], self.get_system_metrics()
    
    def change_tier(self, tier: str):
        """Change processing tier"""
        self.current_tier = tier.lower()
        return self.get_system_metrics()

def create_modern_interface():
    """Create the modern Gradio interface"""
    
    interface = MonoVisionModernInterface()
    
    # Initialize the interface - handle async properly
    try:
        # Try to get current event loop (Gradio might already have one)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in a running event loop, initialize synchronously
                logger.info("🔄 Running in existing event loop, using sync initialization")
                success = interface.initialize_sync()
            else:
                # Event loop exists but not running, use it
                success = loop.run_until_complete(interface.initialize())
        except RuntimeError:
            # No event loop, create a new one
            logger.info("🔄 Creating new event loop for initialization")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(interface.initialize())
            loop.close()
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        success = False
    
    if not success:
        raise Exception("Failed to initialize MonoVision interface")
    
    # Create the Gradio interface with modern theme
    with gr.Blocks(
        title="MonoVision V4 - Ultra-Enhanced AI Vision Interface",
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue="blue",
            secondary_hue="cyan",
            neutral_hue="slate",
            spacing_size="md",
            radius_size="md"
        ).set(
            background_fill_primary=THEME_COLORS["background"],
            background_fill_secondary=THEME_COLORS["surface"],
            border_color_primary=THEME_COLORS["border"],
            color_accent=THEME_COLORS["primary"],
            color_accent_soft=THEME_COLORS["primary"] + "30"
        )
    ) as app:
        
        # Header with logo and status
        with gr.Row():
            # Dynamic header based on available features
            if ENHANCED_FEATURES_AVAILABLE:
                header_title = "🌟 MonoVision V4 Ultra-Enhanced"
                header_subtitle = "Advanced AI Vision & Language Processing • Intelligent Chatbot • Production Dashboard"
                header_badges = """
                    <div style="margin-top: 8px;">
                        <span style="background: {primary}20; color: {primary}; padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600;">
                            ⚡ NEW: Enhanced Object Detection
                        </span>
                        <span style="background: {secondary}20; color: {secondary}; padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; margin-left: 8px;">
                            🎯 Comprehensive Image Analysis
                        </span>
                        <span style="background: {success}20; color: {success}; padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600; margin-left: 8px;">
                            📊 Advanced Technical Details
                        </span>
                    </div>
                """.format(
                    primary=THEME_COLORS['primary'],
                    secondary=THEME_COLORS['secondary'],
                    success=THEME_COLORS['success']
                )
            else:
                header_title = "🌟 MonoVision V4"
                header_subtitle = "Ultra-Enhanced AI Vision & Language Processing • Streaming Output • Smart Caching"
                header_badges = f"""
                    <div style="margin-top: 8px;">
                        <span style="background: {THEME_COLORS['primary']}20; color: {THEME_COLORS['primary']}; padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 600;">
                            ⚡ NEW: Advanced Analysis Pipeline
                        </span>
                    </div>
                """
            
            gr.HTML(f"""
                <div style="text-align: center; padding: 20px; background: {THEME_COLORS['surface']}; border-radius: 12px; margin-bottom: 20px; border: 1px solid {THEME_COLORS['border']};">
                    <h1 style="color: {THEME_COLORS['primary']}; font-size: 32px; margin: 0; font-weight: bold;">
                        {header_title}
                    </h1>
                    <p style="color: {THEME_COLORS['text_secondary']}; margin: 10px 0 0 0; font-size: 16px;">
                        {header_subtitle}
                    </p>
                    {header_badges}
                </div>
            """)
        
        # Main interface layout
        with gr.Row():
            # Left sidebar
            with gr.Column(scale=1, min_width=300):
                gr.HTML(f"""
                    <div style="background: {THEME_COLORS['surface']}; padding: 20px; border-radius: 12px; border: 1px solid {THEME_COLORS['border']}; margin-bottom: 20px;">
                        <h3 style="color: {THEME_COLORS['primary']}; margin: 0 0 15px 0;">⚙️ Settings</h3>
                    </div>
                """)
                
                # Tier selector
                processing_mode = gr.Radio(
                    choices=["fast", "balanced", "rich"],
                    value="fast",
                    label="🎯 Processing Tier",
                    info="Choose your processing mode"
                )
                
                # Enhanced mode recommendation panel (V3 feature)
                mode_recommendation_panel = gr.Markdown(
                    value="**🧠 AI Mode Recommendation:**\n\nType a question to get intelligent mode suggestions",
                    label="💡 Smart Mode Selection"
                )
                
                # Image upload with preview
                image_input = gr.Image(
                    label="🖼️ Upload Image",
                    type="pil",
                    height=200
                )
                
                # Image preview area
                with gr.Group():
                    gr.HTML(f"""
                        <div style="background: {THEME_COLORS['surface']}; padding: 15px; border-radius: 8px; border: 1px solid {THEME_COLORS['border']}; margin: 10px 0;">
                            <h4 style="color: {THEME_COLORS['primary']}; margin: 0 0 10px 0;">🖼️ Image Preview</h4>
                        </div>
                    """)
                    
                    image_preview = gr.Image(
                        label="Current Image",
                        type="pil",
                        height=150,
                        interactive=False,
                        visible=True
                    )
                    
                    image_info = gr.Markdown(
                        value="No image uploaded",
                        label="📋 Image Info"
                    )
                
                # Include objects toggle
                include_objects = gr.Checkbox(
                    label="📦 Include Object Detection",
                    value=True,
                    info="Enable YOLOv8 object detection (works in Balanced/Rich modes)"
                )
                
                # Send image to Gemini toggle
                send_image_to_gemini = gr.Checkbox(
                    label="🎯 Send Image to Gemini API",
                    value=False,
                    info="Send the uploaded image directly to Gemini API for enhanced analysis (Rich mode only)"
                )
                
                # System metrics
                metrics_display = gr.Markdown(
                    value=interface.get_system_metrics(),
                    label="📊 System Status"
                )
                
                # Enhanced production dashboard (V3 feature)
                if ENHANCED_FEATURES_AVAILABLE:
                    # Initialize dashboard with actual metrics
                    initial_dashboard = interface.get_dashboard_summary()
                    dashboard_panel = gr.HTML(
                        value=initial_dashboard,
                        label="🎛️ Advanced Monitoring"
                    )
                else:
                    dashboard_panel = gr.Markdown(
                        value="**📊 Basic Monitoring:**\n\nUpgrade to V3 for advanced dashboard",
                        label="📊 System Monitoring"
                    )
                
                # Technical details panel
                technical_panel = gr.Markdown(
                    value=interface.get_technical_details(),
                    label="🔬 Technical Analysis"
                )
                
                # Interactive overlay display
                interactive_display = gr.HTML(
                    value="Upload an image to see interactive object detection",
                    label="🎯 Interactive Objects"
                )
                
                # Suggested questions panel
                suggested_questions_panel = gr.Markdown(
                    value=interface.get_suggested_questions(),
                    label="💡 Suggested Questions"
                )
            
            # Main workspace
            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="💬 AI Assistant",
                    height=500,
                    show_label=True,
                    container=True,
                    bubble_full_width=False,
                    type="messages"  # Use messages format for role/content dicts (Gradio 5.x)
                )
                
                # Input row
                with gr.Row():
                    text_input = gr.Textbox(
                        label="💭 Your Message",
                        placeholder="Ask about the image or chat with AI...",
                        scale=4,
                        show_label=False
                    )
                    
                    send_btn = gr.Button(
                        "🚀 Send",
                        variant="primary",
                        scale=1
                    )
                
                # Action buttons
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                    refresh_btn = gr.Button("🔄 Refresh Metrics", variant="secondary")
        
        # Event handlers
        def update_mode_recommendation(text_input, image_input):
            """Update mode recommendation based on text input"""
            has_image = image_input is not None
            recommendation = interface.get_mode_recommendation(text_input or "", has_image)
            return recommendation
        
        # Image upload preview - shows uploaded image and info, plus interactive overlay
        def update_image_and_overlay(image):
            """Update image preview and create interactive overlay if objects detected"""
            preview_image, info_text = interface.update_image_preview(image)
            
            # Create interactive overlay if we have stored objects
            if hasattr(interface, 'current_objects') and interface.current_objects and image:
                overlay_html = interface.create_interactive_overlay(image, interface.current_objects)
            else:
                overlay_html = "Upload an image and enable object detection to see interactive overlays"
            
            suggested_questions = interface.get_suggested_questions()
            
            return preview_image, info_text, overlay_html, suggested_questions
        
        image_input.change(
            update_image_and_overlay,
            inputs=[image_input],
            outputs=[image_preview, image_info, interactive_display, suggested_questions_panel]
        )
        
        # Update mode recommendation when text changes
        text_input.change(
            update_mode_recommendation,
            inputs=[text_input, image_input],
            outputs=[mode_recommendation_panel]
        )
        
        def process_and_update_all_streaming(image_input, text_input, processing_mode, include_objects, send_image_to_gemini, chatbot):
            """Process input and update all interface components with streaming"""
            # Use the streaming processing function
            for text_out, image_out, chat_out, metrics_out in interface.process_image_and_text_streaming(
                image_input, text_input, processing_mode, include_objects, send_image_to_gemini, chatbot
            ):
                # Update technical details
                technical_out = interface.get_technical_details()
                
                # Update enhanced dashboard
                dashboard_out = interface.get_dashboard_summary()
                
                # Update interactive overlay if we processed an image with objects
                if include_objects and interface.current_objects and image_input:
                    try:
                        overlay_html = interface.create_interactive_overlay(image_input, interface.current_objects)
                    except Exception as e:
                        overlay_html = f"<div style='color: red;'>Overlay error: {e}</div>"
                else:
                    overlay_html = "Enable object detection to see interactive overlays"
                
                # Update suggested questions
                try:
                    suggested_questions = interface.get_suggested_questions()
                except Exception as e:
                    suggested_questions = "**💡 Suggested Questions:**\n\nError loading suggestions"
                
                # Ensure all outputs are safe strings/types that won't cause file path issues
                safe_technical_out = str(technical_out) if technical_out else "No analysis performed yet"
                safe_overlay_html = str(overlay_html) if overlay_html else "No overlay available"
                safe_suggested_questions = str(suggested_questions) if suggested_questions else "**💡 Suggested Questions:**\n\nNo suggestions available"
                safe_dashboard_out = str(dashboard_out) if dashboard_out else interface.get_dashboard_summary()
                
                # Don't update image_input component to preserve image preview
                yield "", chat_out, metrics_out, safe_technical_out, safe_overlay_html, safe_suggested_questions, safe_dashboard_out

        def process_and_update_all(image_input, text_input, processing_mode, include_objects, send_image_to_gemini, chatbot):
            """Process input and update all interface components (non-streaming fallback)"""
            # Main processing
            text_out, image_out, chat_out, metrics_out = interface.process_image_and_text(
                image_input, text_input, processing_mode, include_objects, send_image_to_gemini, chatbot
            )
            
            # Update technical details
            technical_out = interface.get_technical_details()
            
            # Update enhanced dashboard
            dashboard_out = interface.get_dashboard_summary()
            
            # Update interactive overlay if we processed an image with objects
            if include_objects and interface.current_objects and image_input:
                try:
                    overlay_html = interface.create_interactive_overlay(image_input, interface.current_objects)
                except Exception as e:
                    overlay_html = f"<div style='color: red;'>Overlay error: {e}</div>"
            else:
                overlay_html = "Enable object detection to see interactive overlays"
            
            # Update suggested questions
            try:
                suggested_questions = interface.get_suggested_questions()
            except Exception as e:
                suggested_questions = "**💡 Suggested Questions:**\n\nError loading suggestions"
            
            # Ensure all outputs are safe strings/types that won't cause file path issues
            safe_technical_out = str(technical_out) if technical_out else "No analysis performed yet"
            safe_overlay_html = str(overlay_html) if overlay_html else "No overlay available"
            safe_suggested_questions = str(suggested_questions) if suggested_questions else "**💡 Suggested Questions:**\n\nNo suggestions available"
            safe_dashboard_out = str(dashboard_out) if dashboard_out else interface.get_dashboard_summary()
            
            # Don't update image input component to preserve image preview
            return "", chat_out, metrics_out, safe_technical_out, safe_overlay_html, safe_suggested_questions, safe_dashboard_out
        
        send_btn.click(
            process_and_update_all_streaming,
            inputs=[image_input, text_input, processing_mode, include_objects, send_image_to_gemini, chatbot],
            outputs=[text_input, chatbot, metrics_display, technical_panel, interactive_display, suggested_questions_panel, dashboard_panel]
        )
        
        text_input.submit(
            process_and_update_all_streaming,
            inputs=[image_input, text_input, processing_mode, include_objects, send_image_to_gemini, chatbot],
            outputs=[text_input, chatbot, metrics_display, technical_panel, interactive_display, suggested_questions_panel, dashboard_panel]
        )
        
        def clear_all():
            """Clear all interface components"""
            chat_out, metrics_out = interface.clear_chat()
            overlay_html = "Upload an image to see interactive object detection"
            suggested_questions = interface.get_suggested_questions()
            technical_details = "No analysis performed yet"
            mode_recommendation = "**🧠 AI Mode Recommendation:**\n\nType a question to get intelligent mode suggestions"
            
            # Enhanced dashboard update if available
            dashboard_info = interface.get_dashboard_summary()
            
            return chat_out, metrics_out, overlay_html, suggested_questions, technical_details, mode_recommendation, dashboard_info
        
        clear_btn.click(
            clear_all,
            outputs=[chatbot, metrics_display, interactive_display, suggested_questions_panel, technical_panel, mode_recommendation_panel, dashboard_panel]
        )
        
        refresh_btn.click(
            lambda: interface.get_system_metrics(),
            outputs=[metrics_display]
        )
        
        processing_mode.change(
            interface.change_tier,
            inputs=[processing_mode],
            outputs=[metrics_display]
        )
    
    return app

def find_available_port(start_port=7860, max_attempts=20):
    """Find an available port starting from start_port with robust binding"""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            # Try to bind to the port with SO_REUSEADDR
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('0.0.0.0', port))
                return port
        except OSError:
            continue
    
    # If no port found, return a random high port
    import random
    return random.randint(8000, 9999)

def stop_existing_servers():
    """Stop any existing Python/Gradio servers"""
    import subprocess
    import os
    import time
    
    try:
        # Kill any existing python processes
        if os.name == 'nt':  # Windows
            subprocess.run(['taskkill', '/F', '/IM', 'python.exe', '/T'], 
                         capture_output=True, text=True)
        else:  # Unix-like
            subprocess.run(['pkill', '-f', 'python'], capture_output=True)
        
        print("🔄 Stopped existing Python processes")
        # Add delay to ensure processes are fully terminated
        time.sleep(3)
        print("⏳ Waited for processes to terminate")
        
    except Exception as e:
        print(f"⚠️ Could not stop existing processes: {e}")

if __name__ == "__main__":
    print("🎯 MONOVISION V4 - ULTRA-ENHANCED MODERN INTERFACE")
    print("=" * 60)
    
    if ENHANCED_FEATURES_AVAILABLE:
        print("✅ V3 ENHANCED FEATURES ENABLED:")
        print("   🤖 Enhanced Intelligent Chatbot")
        print("   🧠 AI-Powered Mode Selection")
        print("   📊 Production Dashboard Monitoring")
        print("   � Object-Specific Query Processing")
        print("   💡 Smart Conversation Suggestions")
    else:
        print("⚠️  V3 Enhanced features not available - using V2 base")
    
    print("\n🎨 Interface Features:")
    print("   �🎨 Modern dark theme with VS Code inspired design")
    print("   ⚡ NEW: Streaming output with ChatGPT-like typing effect")
    print("   🖼️ Interactive image previews in chat")
    print("   💾 Smart structured caching with TTL")
    print("\n🏗️ Tiered Processing:")
    print("   🚀 Fast: BLIP + Flan-T5 (~5-10s)")
    print("   ⚖️ Balanced: BLIP + CLIP + Phi-2 (~15-25s)")
    print("   🔬 Rich: BLIP + CLIP + Third-Party API (~10-20s)")
    print("=" * 60)
    
    # Stop any existing servers first
    stop_existing_servers()
    
    # Find an available port
    port = find_available_port(7860)
    print(f"🔍 Found available port: {port}")
    print(f"🌐 Interface URL: http://localhost:{port}")
    print("=" * 60)
    print("🚀 Starting MonoVision V4 server...")
    print("📱 Once models are loaded, you can access the interface!")
    print("⏹️ Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        app = create_modern_interface()
        app.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=True,
            debug=False,
            show_error=True,
            quiet=False,
            prevent_thread_lock=False
        )
    except Exception as e:
        logger.error(f"❌ Failed to launch interface: {e}")
        print(f"❌ Error: {e}")
        print("\n🆘 SERVER LAUNCH FAILED")
        print("🔧 Troubleshooting steps:")
        print("1. Try running: python server_manager.py")
        print("2. Check if other Python processes are running")
        print("3. Try a different port: python server_manager.py start 8080")
        print("4. Run: taskkill /F /IM python.exe /T")

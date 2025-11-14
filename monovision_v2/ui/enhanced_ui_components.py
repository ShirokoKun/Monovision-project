"""
Enhanced UI Components V3 for MonoVision
- Interactive object overlays with bounding boxes
- Image previews in performance panel
- Async processing indicators
- Smart caching visualization
"""

import json
import base64
import logging
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np

logger = logging.getLogger(__name__)

class InteractiveObjectOverlay:
    """Generate interactive clickable object overlays for YOLO detections"""
    
    def __init__(self):
        self.colors = [
            "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FECA57", 
            "#FF9FF3", "#54A0FF", "#5F27CD", "#00D2D3", "#FF9F43"
        ]
        # Enhanced object descriptions for clickable interactions
        self.object_descriptions = {
            "person": ["individual", "human figure", "person", "someone"],
            "car": ["vehicle", "automobile", "car", "motor vehicle"],
            "truck": ["large vehicle", "truck", "delivery vehicle", "commercial vehicle"],
            "bicycle": ["bike", "bicycle", "two-wheeler", "cycling vehicle"],
            "motorcycle": ["motorbike", "motorcycle", "two-wheeler motor vehicle"],
            "laptop": ["computer", "laptop", "portable computer", "notebook computer"],
            "cell phone": ["phone", "mobile device", "smartphone", "cellular device"],
            "book": ["publication", "book", "reading material", "text"],
            "chair": ["seat", "chair", "furniture for sitting", "seating"],
            "table": ["surface", "table", "flat furniture", "work surface"],
            "bottle": ["container", "bottle", "liquid container", "vessel"],
            "cup": ["drinking vessel", "cup", "beverage container", "mug"],
            "bowl": ["dish", "bowl", "container", "serving dish"],
            "banana": ["fruit", "banana", "yellow fruit", "tropical fruit"],
            "apple": ["fruit", "apple", "red fruit", "round fruit"],
            "sandwich": ["food", "sandwich", "meal", "prepared food"],
            "pizza": ["food", "pizza", "Italian dish", "baked food"],
            "cake": ["dessert", "cake", "sweet food", "baked good"],
            "couch": ["furniture", "sofa", "seating furniture", "living room furniture"],
            "bed": ["furniture", "bed", "sleeping furniture", "bedroom furniture"],
            "toilet": ["fixture", "toilet", "bathroom fixture", "sanitary fixture"],
            "tv": ["device", "television", "display", "entertainment device"],
            "remote": ["controller", "remote control", "device controller", "TV remote"],
            "keyboard": ["input device", "keyboard", "computer peripheral", "typing device"],
            "mouse": ["input device", "computer mouse", "pointing device", "PC accessory"],
            "refrigerator": ["appliance", "fridge", "cooling appliance", "kitchen appliance"],
            "microwave": ["appliance", "microwave", "heating appliance", "kitchen device"],
            "oven": ["appliance", "oven", "baking appliance", "cooking device"],
            "sink": ["fixture", "sink", "washing basin", "kitchen/bathroom fixture"],
            "clock": ["timepiece", "clock", "time display", "wall clock"],
            "scissors": ["tool", "scissors", "cutting tool", "office supply"],
            "hair drier": ["appliance", "hair dryer", "styling tool", "personal care device"],
            "toothbrush": ["tool", "toothbrush", "dental hygiene tool", "oral care item"]
        }

class ObjectOverlayGenerator(InteractiveObjectOverlay):
    """Generate interactive object overlays for YOLO detections with clickable functionality"""
        
    def create_interactive_overlay_html(self, image: Image.Image, objects: List[Dict[str, Any]]) -> str:
        """Create interactive HTML overlay with clickable object boxes"""
        try:
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=90)
            image_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            width, height = image.size
            
            # Create interactive HTML with clickable boxes
            html_content = f"""
            <div style="position: relative; display: inline-block; border-radius: 8px; overflow: hidden;">
                <img src="data:image/jpeg;base64,{image_b64}" 
                     style="max-width: 100%; height: auto; display: block;" 
                     id="detection-image" />
                     
                <!-- Object Detection Boxes -->
            """
            
            for i, obj in enumerate(objects):
                bbox = obj.get('bbox', [0, 0, 0, 0])
                confidence = obj.get('confidence', 0.0)
                class_name = obj.get('class', obj.get('name', 'unknown'))
                color = self.colors[i % len(self.colors)]
                
                # Calculate relative positions (as percentages)
                x_percent = (bbox[0] / width) * 100
                y_percent = (bbox[1] / height) * 100
                w_percent = ((bbox[2] - bbox[0]) / width) * 100
                h_percent = ((bbox[3] - bbox[1]) / height) * 100
                
                # Get contextual description
                descriptions = self.object_descriptions.get(class_name, [class_name])
                description = descriptions[0] if descriptions else class_name
                
                # Create clickable overlay box
                html_content += f"""
                <div class="detection-box" 
                     onclick="showObjectDetails('{class_name}', {confidence:.3f}, '{description}', {i})"
                     style="
                        position: absolute;
                        left: {x_percent}%;
                        top: {y_percent}%;
                        width: {w_percent}%;
                        height: {h_percent}%;
                        border: 2px solid {color};
                        background: {color}20;
                        cursor: pointer;
                        transition: all 0.2s ease;
                        border-radius: 4px;
                        backdrop-filter: blur(1px);
                     "
                     onmouseover="this.style.borderWidth='3px'; this.style.background='{color}40'"
                     onmouseout="this.style.borderWidth='2px'; this.style.background='{color}20'"
                     title="Click for details: {class_name} ({confidence:.2f})">
                     
                    <!-- Object Label -->
                    <div style="
                        position: absolute;
                        top: -25px;
                        left: 0;
                        background: {color};
                        color: white;
                        padding: 2px 6px;
                        border-radius: 3px;
                        font-size: 11px;
                        font-weight: bold;
                        white-space: nowrap;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                    ">
                        {class_name} {confidence:.2f}
                    </div>
                </div>
                """
            
            # Add JavaScript for interactivity
            html_content += f"""
            </div>
            
            <!-- Object Details Modal -->
            <div id="object-details-modal" style="
                display: none;
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: #1E1E1E;
                border: 1px solid #3A86FF;
                border-radius: 12px;
                padding: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.5);
                z-index: 1000;
                min-width: 300px;
                color: #E0E0E0;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h3 style="margin: 0; color: #3A86FF;">Object Details</h3>
                    <button onclick="closeObjectDetails()" style="
                        background: none;
                        border: none;
                        color: #A0A0A0;
                        font-size: 18px;
                        cursor: pointer;
                        padding: 5px;
                    ">×</button>
                </div>
                <div id="object-details-content"></div>
            </div>
            
            <!-- Backdrop -->
            <div id="modal-backdrop" style="
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.5);
                z-index: 999;
            " onclick="closeObjectDetails()"></div>
            
            <script>
            function showObjectDetails(className, confidence, description, index) {{
                const modal = document.getElementById('object-details-modal');
                const backdrop = document.getElementById('modal-backdrop');
                const content = document.getElementById('object-details-content');
                
                // Get additional context from object descriptions
                const objectData = {json.dumps(self.object_descriptions)};
                const alternatives = objectData[className] || [className];
                
                // Generate contextual description
                let contextualDesc = '';
                if (className === 'person') {{
                    contextualDesc = 'This appears to be a person in the image. The detection shows a human figure.';
                }} else if (className === 'laptop') {{
                    contextualDesc = 'This is a laptop computer, brand unknown, likely a portable computing device.';
                }} else if (className === 'car') {{
                    contextualDesc = 'This is a vehicle, appears to be a car or automobile.';
                }} else if (className === 'chair') {{
                    contextualDesc = 'This is seating furniture, likely a chair for sitting.';
                }} else if (className === 'table') {{
                    contextualDesc = 'This is a table or flat surface, possibly for working or dining.';
                }} else if (className === 'cell phone') {{
                    contextualDesc = 'This is a mobile device, likely a smartphone or cell phone.';
                }} else if (className === 'book') {{
                    contextualDesc = 'This appears to be a book or publication for reading.';
                }} else {{
                    contextualDesc = `This is a ${{className}}, detected with ${{(confidence * 100).toFixed(1)}}% confidence.`;
                }}
                
                content.innerHTML = `
                    <div style="margin-bottom: 12px;">
                        <strong style="color: #4CC9F0;">Object:</strong> ${{className}}
                    </div>
                    <div style="margin-bottom: 12px;">
                        <strong style="color: #4CC9F0;">Confidence:</strong> ${{(confidence * 100).toFixed(1)}}%
                    </div>
                    <div style="margin-bottom: 12px;">
                        <strong style="color: #4CC9F0;">Description:</strong><br>
                        <span style="color: #E0E0E0; line-height: 1.4;">${{contextualDesc}}</span>
                    </div>
                    <div style="margin-bottom: 12px;">
                        <strong style="color: #4CC9F0;">Alternative Names:</strong><br>
                        <span style="color: #A0A0A0; font-size: 12px;">${{alternatives.join(', ')}}</span>
                    </div>
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #2A2A2A;">
                        <button onclick="askAboutObject('${{className}}')" style="
                            background: #3A86FF;
                            border: none;
                            color: white;
                            padding: 8px 16px;
                            border-radius: 6px;
                            cursor: pointer;
                            margin-right: 10px;
                        ">Ask AI About This</button>
                        <button onclick="closeObjectDetails()" style="
                            background: #2A2A2A;
                            border: 1px solid #444;
                            color: #E0E0E0;
                            padding: 8px 16px;
                            border-radius: 6px;
                            cursor: pointer;
                        ">Close</button>
                    </div>
                `;
                
                modal.style.display = 'block';
                backdrop.style.display = 'block';
            }}
            
            function closeObjectDetails() {{
                document.getElementById('object-details-modal').style.display = 'none';
                document.getElementById('modal-backdrop').style.display = 'none';
            }}
            
            function askAboutObject(className) {{
                // This would integrate with the chat interface
                const questions = {{
                    'person': 'Can you tell me more about the person in this image?',
                    'laptop': 'What can you tell me about this laptop?',
                    'car': 'Describe the vehicle you see in the image.',
                    'chair': 'What type of chair is this?',
                    'table': 'What is on this table?',
                    'cell phone': 'Can you identify the brand or model of this phone?',
                    'book': 'Can you read any text on this book?'
                }};
                
                const question = questions[className] || `Tell me more about this ${{className}}.`;
                
                // Try to populate the text input if it exists
                const textInput = document.querySelector('input[placeholder*="Ask about"], textarea[placeholder*="Ask about"], input[placeholder*="Your Message"], textarea[placeholder*="Your Message"]');
                if (textInput) {{
                    textInput.value = question;
                    textInput.focus();
                }}
                
                closeObjectDetails();
                
                // Show a notification
                showNotification(`Question added: "${{question}}"`);
            }}
            
            function showNotification(message) {{
                const notification = document.createElement('div');
                notification.style.cssText = `
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: #3A86FF;
                    color: white;
                    padding: 12px 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                    z-index: 1001;
                    font-size: 14px;
                    max-width: 300px;
                `;
                notification.textContent = message;
                document.body.appendChild(notification);
                
                setTimeout(() => {{
                    notification.remove();
                }}, 3000);
            }}
            </script>
            """
            
            return html_content
            
        except Exception as e:
            logger.error(f"❌ Error creating interactive overlay: {e}")
            return f"<div style='color: red;'>Error creating interactive overlay: {e}</div>"

    def create_overlay_data(self, image: Image.Image, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create overlay data for object detection visualization"""
        try:
            # Convert image to base64 for web display
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            image_b64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Process object detections
            overlay_objects = []
            for i, obj in enumerate(objects):
                color = self.colors[i % len(self.colors)]
                
                # Extract bounding box coordinates
                bbox = obj.get('bbox', [0, 0, 0, 0])
                confidence = obj.get('confidence', 0.0)
                class_name = obj.get('class', 'unknown')
                
                overlay_obj = {
                    "id": f"obj_{i}",
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": {
                        "x": int(bbox[0]),
                        "y": int(bbox[1]),
                        "width": int(bbox[2] - bbox[0]),
                        "height": int(bbox[3] - bbox[1])
                    },
                    "color": color,
                    "label": f"{class_name} ({confidence:.2f})"
                }
                overlay_objects.append(overlay_obj)
            
            return {
                "image_b64": image_b64,
                "image_size": image.size,
                "objects": overlay_objects,
                "total_objects": len(objects),
                "overlay_markdown": self._generate_overlay_markdown(image.size, overlay_objects),
                "interactive_html": self.create_interactive_overlay_html(image, objects),
                "clickable_objects": self._generate_clickable_summaries(overlay_objects)
            }
            
        except Exception as e:
            logger.error(f"❌ Error creating overlay data: {e}")
            return {"error": str(e)}
    
    def _generate_overlay_markdown(self, image_size: Tuple[int, int], objects: List[Dict]) -> str:
        """Generate Markdown-formatted object overlay for chat display"""
        width, height = image_size
        
        if not objects:
            return "🖼️ **Image Analysis:** No objects detected in the image."
        
        # Create markdown report of detected objects
        overlay_md = f"🎯 **Object Detection Results** ({len(objects)} objects found)\n\n"
        
        for i, obj in enumerate(objects, 1):
            bbox = obj["bbox"]
            position_text = f"at position ({bbox['x']}, {bbox['y']}) • size {bbox['width']}×{bbox['height']}px"
            
            overlay_md += f"**{i}.** `{obj['class']}` - **{obj['confidence']:.2f}%** confidence • {position_text}\n"
        
        overlay_md += f"\n📐 **Image Dimensions:** {width} × {height} pixels"
        
        return overlay_md
    
    def _generate_clickable_summaries(self, objects: List[Dict]) -> List[Dict[str, Any]]:
        """Generate clickable object summaries for interactive use"""
        clickable_summaries = []
        
        for obj in objects:
            class_name = obj["class"]
            confidence = obj["confidence"]
            
            # Get contextual descriptions
            descriptions = self.object_descriptions.get(class_name, [class_name])
            primary_desc = descriptions[0] if descriptions else class_name
            
            # Generate suggested questions
            suggested_questions = self._generate_object_questions(class_name)
            
            clickable_summaries.append({
                "id": obj["id"],
                "class": class_name,
                "confidence": confidence,
                "primary_description": primary_desc,
                "alternative_names": descriptions,
                "suggested_questions": suggested_questions,
                "bbox": obj["bbox"],
                "color": obj["color"]
            })
        
        return clickable_summaries
    
    def _generate_object_questions(self, class_name: str) -> List[str]:
        """Generate contextual questions for each object type"""
        question_templates = {
            "person": [
                "How many people are in the image?",
                "What is this person doing?",
                "Can you describe this person's appearance?",
                "What is the person wearing?"
            ],
            "car": [
                "What type of car is this?",
                "What color is the vehicle?",
                "Can you identify the car's brand or model?",
                "Where is the car located?"
            ],
            "laptop": [
                "What brand is this laptop?",
                "Is the laptop open or closed?",
                "What can you see on the laptop screen?",
                "What type of laptop is this?"
            ],
            "chair": [
                "What type of chair is this?",
                "What material is the chair made of?",
                "Is anyone sitting in the chair?",
                "Where is the chair positioned?"
            ],
            "table": [
                "What is on the table?",
                "What type of table is this?",
                "What material is the table made of?",
                "How many items are on the table?"
            ],
            "cell phone": [
                "What type of phone is this?",
                "Can you identify the phone's brand?",
                "Is the phone screen on or off?",
                "Where is the phone positioned?"
            ],
            "book": [
                "Can you read the book's title?",
                "What type of book is this?",
                "Is the book open or closed?",
                "Can you see any text on the book?"
            ],
            "bottle": [
                "What type of bottle is this?",
                "Is the bottle full or empty?",
                "Can you see any labels on the bottle?",
                "What might be inside the bottle?"
            ],
            "cup": [
                "What type of cup is this?",
                "Is there anything in the cup?",
                "What material is the cup made of?",
                "Can you see any designs on the cup?"
            ]
        }
        
        # Return specific questions or generic ones
        return question_templates.get(class_name, [
            f"Tell me more about this {class_name}.",
            f"What can you observe about the {class_name}?",
            f"Describe the {class_name} in detail.",
            f"What is notable about this {class_name}?"
        ])

class AIQuestionHandler:
    """Handle contextual AI questions about uploaded images and detected objects"""
    
    def __init__(self):
        self.question_categories = {
            "counting": ["how many", "count", "number of"],
            "identification": ["what is", "what are", "identify", "what type"],
            "location": ["where", "position", "located", "placement"],
            "description": ["describe", "tell me about", "appearance", "looks like"],
            "activity": ["doing", "action", "activity", "happening"],
            "color": ["color", "colored", "what color"],
            "size": ["size", "big", "small", "large", "tiny"],
            "material": ["made of", "material", "fabric", "wood", "metal"],
            "brand": ["brand", "manufacturer", "company", "logo"],
            "text": ["text", "writing", "words", "read", "says"]
        }
    
    def categorize_question(self, question: str) -> str:
        """Categorize user question to provide better context for AI"""
        question_lower = question.lower()
        
        for category, keywords in self.question_categories.items():
            if any(keyword in question_lower for keyword in keywords):
                return category
        
        return "general"
    
    def enhance_question_with_context(self, question: str, vision_results: Dict[str, Any]) -> str:
        """Enhance user question with relevant vision context"""
        category = self.categorize_question(question)
        
        # Extract relevant context from vision results
        caption = vision_results.get('caption', '')
        objects = vision_results.get('objects', [])
        clip_keywords = vision_results.get('clip_keywords', [])
        
        # Build enhanced context based on question category
        context_parts = []
        
        if category == "counting":
            # For counting questions, focus on object detection
            object_counts = {}
            for obj in objects:
                obj_name = obj.get('class', obj.get('name', 'unknown'))
                object_counts[obj_name] = object_counts.get(obj_name, 0) + 1
            
            if object_counts:
                context_parts.append(f"Objects detected: {', '.join([f'{count} {name}' for name, count in object_counts.items()])}")
        
        elif category == "identification":
            # For identification questions, provide object and caption info
            if objects:
                object_names = [obj.get('class', obj.get('name', 'unknown')) for obj in objects]
                context_parts.append(f"Detected objects: {', '.join(set(object_names))}")
            if caption:
                context_parts.append(f"Scene description: {caption}")
        
        elif category == "location" or category == "description":
            # For location/description questions, use caption and keywords
            if caption:
                context_parts.append(f"Visual description: {caption}")
            if clip_keywords:
                context_parts.append(f"Scene characteristics: {', '.join(clip_keywords[:5])}")
        
        elif category == "activity":
            # For activity questions, focus on caption and human objects
            if caption:
                context_parts.append(f"Scene context: {caption}")
            
            # Look for people in objects
            people_objects = [obj for obj in objects if 'person' in obj.get('class', '').lower()]
            if people_objects:
                context_parts.append(f"People detected: {len(people_objects)} person(s)")
        
        elif category == "color" or category == "material" or category == "brand":
            # For specific attribute questions, provide detailed context
            if caption:
                context_parts.append(f"Visual details: {caption}")
            if clip_keywords:
                relevant_keywords = [kw for kw in clip_keywords if any(attr in kw.lower() for attr in ['color', 'material', 'brand', 'style'])]
                if relevant_keywords:
                    context_parts.append(f"Visual attributes: {', '.join(relevant_keywords)}")
        
        # Combine original question with context
        if context_parts:
            enhanced_question = f"{question}\n\nContext from image analysis:\n" + "\n".join([f"• {part}" for part in context_parts])
        else:
            enhanced_question = question
        
        return enhanced_question
    
    def generate_followup_questions(self, question: str, objects: List[Dict[str, Any]]) -> List[str]:
        """Generate relevant follow-up questions based on current question and detected objects"""
        category = self.categorize_question(question)
        object_types = list(set([obj.get('class', obj.get('name', 'unknown')) for obj in objects]))
        
        followup_questions = []
        
        if category == "counting":
            if len(object_types) > 1:
                followup_questions.extend([
                    f"Are there more {object_types[0]}s than {object_types[1]}s?",
                    "What is the total number of objects in the image?",
                    "Which objects appear most frequently?"
                ])
        
        elif category == "identification":
            if object_types:
                followup_questions.extend([
                    f"Can you describe the {object_types[0]} in more detail?",
                    "What is the overall setting or environment?",
                    "Are there any notable brands or labels visible?"
                ])
        
        elif category == "description":
            followup_questions.extend([
                "What colors are most prominent in the image?",
                "What is the lighting like in this scene?",
                "Does this appear to be indoors or outdoors?",
                "What is the overall mood or atmosphere?"
            ])
        
        elif category == "activity":
            followup_questions.extend([
                "What might happen next in this scene?",
                "Is this a typical everyday activity?",
                "What tools or equipment are being used?",
                "Is anyone interacting with objects?"
            ])
        
        # Add object-specific questions
        for obj_type in object_types[:2]:  # Limit to first 2 object types
            if obj_type in ["person", "people"]:
                followup_questions.append("What is the person wearing?")
                followup_questions.append("What expression does the person have?")
            elif obj_type in ["car", "vehicle"]:
                followup_questions.append("What color is the vehicle?")
                followup_questions.append("Is the vehicle moving or parked?")
            elif obj_type in ["laptop", "computer"]:
                followup_questions.append("Is the laptop screen visible?")
                followup_questions.append("What might the person be working on?")
        
        return followup_questions[:5]  # Return top 5 questions

class SmartImageInteraction:
    """Main class for managing smart image interactions"""
    
    def __init__(self):
        self.overlay_generator = ObjectOverlayGenerator()
        self.question_handler = AIQuestionHandler()
        self.interaction_history = []
    
    def process_image_upload(self, image: Image.Image, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process new image upload and prepare for interaction"""
        # Generate overlay data
        overlay_data = self.overlay_generator.create_overlay_data(image, objects)
        
        # Generate initial suggestions
        suggested_questions = self._generate_initial_questions(objects)
        
        # Store interaction context
        interaction_context = {
            "image_hash": hash(image.tobytes()),
            "objects": objects,
            "overlay_data": overlay_data,
            "suggested_questions": suggested_questions,
            "interaction_count": 0
        }
        
        self.interaction_history.append(interaction_context)
        
        return {
            "overlay_data": overlay_data,
            "suggested_questions": suggested_questions,
            "interaction_ready": True
        }
    
    def handle_question(self, question: str, vision_results: Dict[str, Any], objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle user question about the image"""
        # Enhance question with context
        enhanced_question = self.question_handler.enhance_question_with_context(question, vision_results)
        
        # Generate follow-up questions
        followup_questions = self.question_handler.generate_followup_questions(question, objects)
        
        # Categorize question for better processing
        question_category = self.question_handler.categorize_question(question)
        
        return {
            "enhanced_question": enhanced_question,
            "followup_questions": followup_questions,
            "question_category": question_category,
            "processing_hints": self._get_processing_hints(question_category)
        }
    
    def _generate_initial_questions(self, objects: List[Dict[str, Any]]) -> List[str]:
        """Generate initial suggested questions based on detected objects"""
        if not objects:
            return [
                "What do you see in this image?",
                "Describe the overall scene.",
                "What is the main subject of this image?",
                "What colors are most prominent?"
            ]
        
        object_types = list(set([obj.get('class', obj.get('name', 'unknown')) for obj in objects]))
        questions = []
        
        # Add counting questions if multiple objects
        if len(objects) > 1:
            questions.append(f"How many objects are in this image?")
            if len(object_types) > 1:
                questions.append(f"How many {object_types[0]}s are there?")
        
        # Add identification questions
        questions.append("What is the main focus of this image?")
        
        # Add object-specific questions
        for obj_type in object_types[:2]:
            if obj_type == "person":
                questions.append("What is the person doing?")
                questions.append("What is the person wearing?")
            elif obj_type in ["car", "vehicle"]:
                questions.append("What type of vehicle is this?")
            elif obj_type in ["laptop", "computer"]:
                questions.append("What is displayed on the screen?")
            elif obj_type in ["book", "document"]:
                questions.append("Can you read any text?")
        
        # Add general questions
        questions.extend([
            "Is this indoors or outdoors?",
            "What is the lighting like?",
            "What is the overall mood of this image?"
        ])
        
        return questions[:6]  # Return top 6 questions
    
    def _get_processing_hints(self, question_category: str) -> Dict[str, Any]:
        """Get processing hints for AI based on question category"""
        hints = {
            "counting": {
                "focus_areas": ["object_detection", "spatial_analysis"],
                "response_style": "precise_numerical",
                "include_confidence": True
            },
            "identification": {
                "focus_areas": ["object_detection", "scene_classification", "caption_analysis"],
                "response_style": "descriptive_detailed",
                "include_confidence": False
            },
            "location": {
                "focus_areas": ["spatial_analysis", "scene_understanding"],
                "response_style": "spatial_descriptive",
                "include_confidence": False
            },
            "description": {
                "focus_areas": ["caption_analysis", "visual_attributes", "scene_understanding"],
                "response_style": "rich_descriptive",
                "include_confidence": False
            },
            "activity": {
                "focus_areas": ["human_pose", "scene_context", "temporal_inference"],
                "response_style": "narrative",
                "include_confidence": False
            },
            "color": {
                "focus_areas": ["color_analysis", "visual_attributes"],
                "response_style": "specific_attributes",
                "include_confidence": False
            },
            "brand": {
                "focus_areas": ["text_detection", "logo_recognition", "fine_details"],
                "response_style": "specific_identification",
                "include_confidence": True
            }
        }
        
        return hints.get(question_category, {
            "focus_areas": ["general_analysis"],
            "response_style": "conversational",
            "include_confidence": False
        })

class PerformancePanelUI:
    """Enhanced performance panel with image previews and cache visualization"""
    
    def __init__(self):
        self.thumbnail_size = (120, 90)
    
    def create_performance_panel(self, processing_results: Dict[str, Any], cache_stats: Dict[str, Any]) -> str:
        """Create enhanced performance panel in Markdown format"""
        try:
            # Extract processing metrics
            processing_time = processing_results.get('processing_time', 0)
            blip_time = processing_results.get('vision_metrics', {}).get('blip_time', 0)
            clip_time = processing_results.get('vision_metrics', {}).get('clip_time', 0)
            yolo_time = processing_results.get('vision_metrics', {}).get('yolo_time', 0)
            
            # Cache metrics
            hit_rate = cache_stats.get('performance', {}).get('hit_rate_percent', 0)
            cache_size = cache_stats.get('ram_cache', {}).get('results', {}).get('size_mb', 0)
            
            # Create markdown performance panel
            panel_md = f"""⚡ **Performance Metrics** • Cache Hit: {hit_rate:.1f}%

**Vision Processing Times:**
• **BLIP Caption:** {blip_time:.2f}s
• **CLIP Embeddings:** {clip_time:.2f}s  
• **YOLO Detection:** {yolo_time:.2f}s
• **Total Processing:** {processing_time:.2f}s

**Cache Performance:**
💾 Cache Size: {cache_size:.1f}MB | 🎯 Hit Rate: {hit_rate:.1f}%"""
            
            return panel_md
            
        except Exception as e:
            logger.error(f"❌ Error creating performance panel: {e}")
            return f"**Performance Panel Error:** {e}"

class AsyncProcessingUI:
    """UI components for async processing visualization"""
    
    def create_processing_indicator(self, stage: str, progress: float = 0) -> str:
        """Create animated processing indicator in Markdown format"""
        stages = {
            "initializing": {"icon": "🚀", "text": "Initializing models..."},
            "vision": {"icon": "👁️", "text": "Processing vision..."},
            "blip": {"icon": "📝", "text": "Generating caption..."},
            "clip": {"icon": "🔗", "text": "Analyzing semantics..."},
            "yolo": {"icon": "📦", "text": "Detecting objects..."},
            "fusion": {"icon": "🔄", "text": "Fusing results..."},
            "generation": {"icon": "🎯", "text": "Generating response..."},
            "complete": {"icon": "✅", "text": "Processing complete!"}
        }
        
        stage_info = stages.get(stage, {"icon": "⏳", "text": "Processing..."})
        progress_bar = "█" * int(progress / 10) + "░" * (10 - int(progress / 10))
        
        return f"{stage_info['icon']} **{stage_info['text']}**\n`{progress_bar}` {progress:.1f}%"

class CacheVisualizationUI:
    """Cache visualization components"""
    
    def create_cache_panel(self, cache_stats: Dict[str, Any]) -> str:
        """Create cache visualization panel in Markdown format"""
        try:
            ram_stats = cache_stats.get('ram_cache', {})
            db_stats = cache_stats.get('database', {})
            perf_stats = cache_stats.get('performance', {})
            
            hit_rate = perf_stats.get('hit_rate_percent', 0)
            total_size = sum([
                ram_stats.get('results', {}).get('size_mb', 0),
                ram_stats.get('embeddings', {}).get('size_mb', 0),
                ram_stats.get('images', {}).get('size_mb', 0)
            ])
            
            status_emoji = "🟢" if hit_rate > 50 else "🟡" if hit_rate > 0 else "🔴"
            
            cache_md = f"""💾 **Cache Status** {status_emoji}

**Performance Metrics:**
• **Hit Rate:** {hit_rate:.1f}%
• **RAM Usage:** {total_size:.1f}MB  
• **DB Entries:** {db_stats.get('results_count', 0)}

**Memory Breakdown:**
• Results: {ram_stats.get('results', {}).get('entries', 0)} entries
• Embeddings: {ram_stats.get('embeddings', {}).get('entries', 0)} entries  
• Images: {ram_stats.get('images', {}).get('entries', 0)} entries"""
            
            return cache_md
            
        except Exception as e:
            logger.error(f"❌ Error creating cache panel: {e}")
            return f"**Cache Panel Error:** {e}"

class FusionVisualizationUI:
    """Visualization for YOLOv8 + BLIP fusion"""
    
    def create_fusion_comparison(self, blip_caption: str, yolo_objects: List[str], fused_result: str) -> str:
        """Create fusion comparison visualization in Markdown format"""
        objects_text = ', '.join(yolo_objects) if yolo_objects else 'No objects detected'
        
        fusion_md = f"""🔄 **Vision Fusion Analysis**

**📝 BLIP Caption:**
> {blip_caption}

**📦 YOLO Objects:**
> {objects_text}

**🎯 Fused Result:**
> {fused_result}"""
        
        return fusion_md

def create_thumbnail(image: Image.Image, size: Tuple[int, int] = (120, 90)) -> str:
    """Create base64 thumbnail for UI display"""
    try:
        thumbnail = image.copy()
        thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
        
        buffered = io.BytesIO()
        thumbnail.save(buffered, format="JPEG", quality=80)
        thumbnail_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        return thumbnail_b64
        
    except Exception as e:
        logger.error(f"❌ Error creating thumbnail: {e}")
        return ""

def enhance_gradio_css() -> str:
    """Enhanced CSS for MonoVision V3 UI"""
    return """
    <style>
    .performance-panel-v3 {
        transition: all 0.3s ease;
    }
    .performance-panel-v3:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.4) !important;
    }
    
    .object-overlay-container {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .cache-panel-v3 {
        transition: all 0.3s ease;
    }
    .cache-panel-v3:hover {
        border-color: #64B5F6;
    }
    
    .fusion-comparison-v3 {
        transition: all 0.3s ease;
    }
    .fusion-comparison-v3:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }
    
    .metric-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0;
    }
    
    .async-processing-indicator {
        margin: 8px 0;
    }
    
    @media (max-width: 768px) {
        .performance-panel-v3 > div:first-child {
            flex-direction: column !important;
            align-items: flex-start !important;
        }
        
        .metrics-grid {
            grid-template-columns: 1fr !important;
        }
    }
    </style>
    """

"""
YOLOv8 Object Detection for MonoVision V3
Integrated object detection with result caching and GPU optimization
"""

import torch
import cv2
import numpy as np
from PIL import Image
import io
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("⚠️ Ultralytics YOLO not available. Install with: pip install ultralytics")

logger = logging.getLogger(__name__)

@dataclass
class DetectedObject:
    """Detected object information"""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]  # center coordinates
    area: int

@dataclass
class ObjectDetectionResult:
    """Complete object detection result"""
    objects: List[DetectedObject]
    detection_time: float
    total_objects: int
    high_confidence_objects: int  # confidence > 0.7
    primary_objects: List[str]  # most prominent object classes
    scene_description: str

class YOLOv8Detector:
    """
    YOLOv8 Object Detection for MonoVision
    - GPU-optimized detection
    - Confidence filtering 
    - Result caching integration
    - Scene understanding
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = None
        self.is_initialized = False
        
        # YOLO class names (COCO dataset)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        logger.info(f"🎯 YOLOv8 Detector initialized (confidence: {confidence_threshold})")
    
    async def initialize(self):
        """Initialize YOLOv8 model"""
        if not YOLO_AVAILABLE:
            logger.error("❌ Ultralytics YOLO not available")
            return False
        
        try:
            # Detect device
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"🚀 YOLO using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = "cpu"
                logger.info("💻 YOLO using CPU")
            
            # Load model
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # Warm up model with a dummy prediction
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_image, verbose=False)
            
            self.is_initialized = True
            logger.info("✅ YOLOv8 model loaded and warmed up")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing YOLOv8: {e}")
            self.is_initialized = False
            return False
    
    def preprocess_image(self, image_data: bytes) -> np.ndarray:
        """Preprocess image for YOLO detection"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            return image_array
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing image: {e}")
            return None
    
    async def detect_objects(self, image_data: bytes) -> Optional[ObjectDetectionResult]:
        """Detect objects in image"""
        if not self.is_initialized:
            if not await self.initialize():
                return None
        
        try:
            start_time = time.time()
            
            # Preprocess image
            image_array = self.preprocess_image(image_data)
            if image_array is None:
                return None
            
            # Run detection
            results = self.model(image_array, conf=self.confidence_threshold, verbose=False)
            
            # Parse results
            detected_objects = []
            
            if results and len(results) > 0:
                result = results[0]  # First image
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Calculate center and area
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Get class name
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        
                        detected_object = DetectedObject(
                            class_name=class_name,
                            confidence=float(conf),
                            bbox=(x1, y1, x2, y2),
                            center=(center_x, center_y),
                            area=area
                        )
                        
                        detected_objects.append(detected_object)
            
            detection_time = time.time() - start_time
            
            # Analyze results
            high_confidence_objects = sum(1 for obj in detected_objects if obj.confidence > 0.7)
            
            # Get primary objects (most confident and largest)
            if detected_objects:
                # Sort by confidence * area (prominence)
                sorted_objects = sorted(detected_objects, 
                                      key=lambda x: x.confidence * x.area, 
                                      reverse=True)
                primary_objects = list(set([obj.class_name for obj in sorted_objects[:3]]))
            else:
                primary_objects = []
            
            # Generate scene description
            scene_description = self._generate_scene_description(detected_objects)
            
            result = ObjectDetectionResult(
                objects=detected_objects,
                detection_time=detection_time,
                total_objects=len(detected_objects),
                high_confidence_objects=high_confidence_objects,
                primary_objects=primary_objects,
                scene_description=scene_description
            )
            
            logger.info(f"🎯 Detected {len(detected_objects)} objects in {detection_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error during object detection: {e}")
            return None
    
    def _generate_scene_description(self, objects: List[DetectedObject]) -> str:
        """Generate natural language scene description"""
        if not objects:
            return "No objects detected in the scene."
        
        # Count objects by class
        object_counts = {}
        for obj in objects:
            object_counts[obj.class_name] = object_counts.get(obj.class_name, 0) + 1
        
        # Sort by count and confidence
        sorted_items = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Generate description
        if len(sorted_items) == 1:
            class_name, count = sorted_items[0]
            if count == 1:
                return f"The scene contains a {class_name}."
            else:
                return f"The scene contains {count} {class_name}{'s' if count > 1 else ''}."
        
        # Multiple object types
        description_parts = []
        
        for i, (class_name, count) in enumerate(sorted_items):
            if i < 3:  # Limit to top 3 object types
                if count == 1:
                    description_parts.append(f"a {class_name}")
                else:
                    description_parts.append(f"{count} {class_name}{'s' if count > 1 else ''}")
        
        if len(description_parts) == 2:
            description = f"The scene contains {description_parts[0]} and {description_parts[1]}."
        elif len(description_parts) > 2:
            description = f"The scene contains {', '.join(description_parts[:-1])}, and {description_parts[-1]}."
        else:
            description = f"The scene contains {description_parts[0]}."
        
        # Add confidence note
        high_conf_count = sum(1 for obj in objects if obj.confidence > 0.8)
        if high_conf_count == len(objects):
            description += " All detections are high confidence."
        elif high_conf_count > 0:
            description += f" {high_conf_count} detection{'s' if high_conf_count > 1 else ''} {'are' if high_conf_count > 1 else 'is'} high confidence."
        
        return description
    
    def get_object_summary(self, detection_result: ObjectDetectionResult) -> Dict[str, Any]:
        """Get summary of detected objects for caching"""
        if not detection_result or not detection_result.objects:
            return {"objects": [], "summary": "No objects detected"}
        
        # Create object summary
        object_summary = []
        for obj in detection_result.objects:
            object_summary.append({
                "class": obj.class_name,
                "confidence": round(obj.confidence, 3),
                "bbox": obj.bbox,
                "area": obj.area
            })
        
        return {
            "objects": object_summary,
            "total_count": detection_result.total_objects,
            "high_confidence_count": detection_result.high_confidence_objects,
            "primary_objects": detection_result.primary_objects,
            "scene_description": detection_result.scene_description,
            "detection_time": round(detection_result.detection_time, 3)
        }
    
    def create_annotated_image(self, image_data: bytes, detection_result: ObjectDetectionResult) -> Optional[bytes]:
        """Create annotated image with bounding boxes"""
        try:
            # Convert to OpenCV format
            image = Image.open(io.BytesIO(image_data))
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Draw bounding boxes
            for obj in detection_result.objects:
                x1, y1, x2, y2 = obj.bbox
                
                # Color based on confidence
                if obj.confidence > 0.8:
                    color = (0, 255, 0)  # Green for high confidence
                elif obj.confidence > 0.6:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 0, 255)  # Red for low confidence
                
                # Draw bounding box
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{obj.class_name}: {obj.confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Label background
                cv2.rectangle(image_cv, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                
                # Label text
                cv2.putText(image_cv, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Convert back to bytes
            image_pil = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            image_pil.save(buffer, format='JPEG', quality=95)
            
            return buffer.getvalue()
            
        except Exception as e:
            logger.error(f"❌ Error creating annotated image: {e}")
            return None

# Global detector instance
_detector_instance = None

async def get_detector() -> YOLOv8Detector:
    """Get or create YOLOv8 detector instance"""
    global _detector_instance
    
    if _detector_instance is None:
        _detector_instance = YOLOv8Detector()
        await _detector_instance.initialize()
    
    return _detector_instance

async def detect_objects_in_image(image_data: bytes) -> Optional[ObjectDetectionResult]:
    """Convenience function for object detection"""
    detector = await get_detector()
    return await detector.detect_objects(image_data)

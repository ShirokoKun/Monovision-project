"""
Image Preprocessor for MonoVision V3
Handles image normalization, resizing, and optional filters according to the pipeline architecture
"""

import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Image preprocessor implementing the first stage of the MonoVision pipeline:
    - Resize / Normalize
    - Optional Filters  
    - Prepare for async pipeline
    """
    
    def __init__(self):
        self.target_size = (384, 384)  # Optimal for BLIP and CLIP
        self.max_size = (1024, 1024)   # Maximum size to prevent memory issues
        
        # Standard normalization for vision models
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Transform for tensor conversion
        self.to_tensor = transforms.ToTensor()
        
        logger.info("🔧 Image Preprocessor initialized")
    
    def preprocess_image(
        self, 
        image: Image.Image,
        target_size: Optional[Tuple[int, int]] = None,
        enhance_quality: bool = False,
        apply_filters: bool = False
    ) -> Dict[str, Any]:
        """
        Preprocess image according to pipeline architecture
        
        Args:
            image: Input PIL Image
            target_size: Target size for resizing (default: 384x384)
            enhance_quality: Whether to apply quality enhancement
            apply_filters: Whether to apply optional filters
            
        Returns:
            Dict containing processed image and metadata
        """
        try:
            logger.info(f"🔧 Preprocessing image: {image.size} -> {target_size or self.target_size}")
            
            # Store original metadata
            original_size = image.size
            original_mode = image.mode
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.info(f"🔧 Converted image from {original_mode} to RGB")
            
            # Apply quality enhancement if requested
            if enhance_quality:
                image = self._enhance_image_quality(image)
                logger.info("✨ Applied quality enhancement")
            
            # Apply optional filters if requested
            if apply_filters:
                image = self._apply_optional_filters(image)
                logger.info("🎨 Applied optional filters")
            
            # Resize image intelligently
            processed_image = self._smart_resize(image, target_size or self.target_size)
            
            # Create tensor version for models that need it
            image_tensor = self._to_tensor_normalized(processed_image)
            
            # Generate preprocessing metadata
            metadata = {
                "original_size": original_size,
                "processed_size": processed_image.size,
                "original_mode": original_mode,
                "processed_mode": processed_image.mode,
                "enhancement_applied": enhance_quality,
                "filters_applied": apply_filters,
                "resize_method": "smart_resize",
                "preprocessing_quality": self._assess_quality(processed_image)
            }
            
            logger.info(f"✅ Image preprocessing complete: {original_size} -> {processed_image.size}")
            
            return {
                "image": processed_image,           # PIL Image for BLIP/YOLO
                "image_tensor": image_tensor,       # Normalized tensor for direct model use
                "metadata": metadata,
                "original_image": image if not enhance_quality and not apply_filters else None
            }
            
        except Exception as e:
            logger.error(f"❌ Image preprocessing failed: {e}")
            raise Exception(f"Image preprocessing error: {e}")
    
    def _smart_resize(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """
        Intelligently resize image maintaining aspect ratio
        """
        original_width, original_height = image.size
        target_width, target_height = target_size
        
        # Check if image is too large and needs downsizing first
        if original_width > self.max_size[0] or original_height > self.max_size[1]:
            # Downsize to maximum while maintaining aspect ratio
            image.thumbnail(self.max_size, Image.Resampling.LANCZOS)
            logger.info(f"🔧 Downsized large image to {image.size}")
        
        # Calculate aspect ratios
        original_ratio = original_width / original_height
        target_ratio = target_width / target_height
        
        if abs(original_ratio - target_ratio) < 0.1:
            # Aspect ratios are close, direct resize
            return image.resize(target_size, Image.Resampling.LANCZOS)
        else:
            # Maintain aspect ratio with padding or cropping
            if original_ratio > target_ratio:
                # Image is wider, fit by height
                new_height = target_height
                new_width = int(new_height * original_ratio)
            else:
                # Image is taller, fit by width
                new_width = target_width
                new_height = int(new_width / original_ratio)
            
            # Resize maintaining aspect ratio
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with target size and paste the resized image
            new_image = Image.new('RGB', target_size, (128, 128, 128))  # Gray background
            
            # Calculate position to center the image
            x = (target_width - new_width) // 2
            y = (target_height - new_height) // 2
            
            # Paste the resized image onto the centered background
            new_image.paste(image, (x, y))
            
            return new_image
    
    def _enhance_image_quality(self, image: Image.Image) -> Image.Image:
        """
        Apply quality enhancements to improve vision model performance
        """
        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        # Enhance sharpness slightly
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.05)
        
        # Enhance color saturation slightly
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.05)
        
        return image
    
    def _apply_optional_filters(self, image: Image.Image) -> Image.Image:
        """
        Apply optional filters for better processing
        """
        # Apply a slight unsharp mask for edge enhancement
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=1))
        
        return image
    
    def _to_tensor_normalized(self, image: Image.Image) -> torch.Tensor:
        """
        Convert PIL image to normalized tensor
        """
        tensor = self.to_tensor(image)
        return self.normalize(tensor)
    
    def _assess_quality(self, image: Image.Image) -> float:
        """
        Assess the quality of the processed image
        Returns a score between 0 and 1
        """
        try:
            # Convert to grayscale for analysis
            gray = image.convert('L')
            gray_array = np.array(gray)
            
            # Calculate image sharpness using Laplacian variance
            laplacian_var = np.var(np.array(gray.filter(ImageFilter.Kernel((3, 3), [-1, -1, -1, -1, 8, -1, -1, -1, -1]))))
            
            # Calculate contrast
            contrast = gray_array.std()
            
            # Normalize and combine metrics
            sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize to 0-1
            contrast_score = min(contrast / 127.0, 1.0)        # Normalize to 0-1
            
            # Combined quality score
            quality_score = (sharpness_score * 0.6 + contrast_score * 0.4)
            
            return min(max(quality_score, 0.0), 1.0)  # Clamp to 0-1
            
        except Exception as e:
            logger.warning(f"⚠️ Could not assess image quality: {e}")
            return 0.5  # Default medium quality
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """
        Get preprocessing statistics and configuration
        """
        return {
            "target_size": self.target_size,
            "max_size": self.max_size,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "supported_modes": ["RGB", "RGBA", "L", "P"],
            "enhancement_available": True,
            "filters_available": True
        }

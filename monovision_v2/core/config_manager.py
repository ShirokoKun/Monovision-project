"""
Configuration Manager for MonoVision V2
Centralized configuration for the concrete tiered architecture
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class VisionConfig:
    """Vision models configuration"""
    blip_model: str = "Salesforce/blip-image-captioning-base"
    clip_model: str = "openai/clip-vit-base-patch32"
    yolo_model: str = "yolov8n.pt"
    image_size: int = 384  # Optimal size for BLIP/CLIP
    use_fp16: bool = True
    gpu_resident: bool = True

@dataclass
class LanguageConfig:
    """Language models configuration for tiered processing"""
    # Tier 1: Fast mode
    flan_model: str = "google/flan-t5-small"
    flan_max_tokens: int = 50  # Increased from 20 for better responses
    flan_gpu_resident: bool = True
    
    # Tier 2: Balanced mode  
    phi2_model: str = "microsoft/phi-2"
    phi2_max_tokens: int = 50
    phi2_quantization: str = "4bit"
    phi2_mixed_precision: bool = True
    
    # Tier 3: Rich mode
    mistral_model: str = "mistralai/Mistral-7B-Instruct-v0.1"
    mistral_max_tokens: int = 100
    mistral_cpu_offload: bool = True
    mistral_offload_ratio: float = 0.6  # 60% to CPU, 40% on GPU

@dataclass
class ResourceConfig:
    """Resource allocation configuration"""
    gpu_memory_budget_gb: float = 4.0
    gpu_memory_overhead_mb: float = 800
    cpu_memory_budget_gb: float = 16.0
    enable_model_offloading: bool = True
    aggressive_cleanup: bool = True
    cache_size_mb: float = 2048  # 2GB cache

@dataclass
class ProcessingConfig:
    """Processing pipeline configuration"""
    default_mode: str = "balanced"  # fast, balanced, rich
    auto_mode_detection: bool = True
    enable_caching: bool = True
    cache_ttl_minutes: int = 60
    max_concurrent_requests: int = 2
    timeout_seconds: int = 120

@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 7864  # New port for V2
    debug: bool = False
    cors_enabled: bool = True
    max_upload_size_mb: int = 50

class ConfigManager:
    """
    Centralized configuration management for MonoVision V2
    Handles the concrete architecture settings and environment variables
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "monovision_v2_config.json"
        
        # Initialize configuration
        self.vision = VisionConfig()
        self.language = LanguageConfig()
        self.resources = ResourceConfig()
        self.processing = ProcessingConfig()
        self.server = ServerConfig()
        
        # Load configuration from file and environment
        self._load_config()
        self._load_environment_variables()
        
        logger.info("⚙️ Configuration Manager initialized")
    
    def _load_config(self):
        """Load configuration from JSON file"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update configurations
                if 'vision' in config_data:
                    self.vision = VisionConfig(**config_data['vision'])
                if 'language' in config_data:
                    self.language = LanguageConfig(**config_data['language'])
                if 'resources' in config_data:
                    self.resources = ResourceConfig(**config_data['resources'])
                if 'processing' in config_data:
                    self.processing = ProcessingConfig(**config_data['processing'])
                if 'server' in config_data:
                    self.server = ServerConfig(**config_data['server'])
                
                logger.info(f"📄 Configuration loaded from {self.config_path}")
            except Exception as e:
                logger.warning(f"⚠️ Error loading config file: {e}, using defaults")
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        # Vision settings
        if os.getenv('MV2_VISION_IMAGE_SIZE'):
            self.vision.image_size = int(os.getenv('MV2_VISION_IMAGE_SIZE'))
        if os.getenv('MV2_VISION_USE_FP16'):
            self.vision.use_fp16 = os.getenv('MV2_VISION_USE_FP16').lower() == 'true'
        
        # Language settings
        if os.getenv('MV2_LANG_FLAN_MAX_TOKENS'):
            self.language.flan_max_tokens = int(os.getenv('MV2_LANG_FLAN_MAX_TOKENS'))
        if os.getenv('MV2_LANG_PHI2_MAX_TOKENS'):
            self.language.phi2_max_tokens = int(os.getenv('MV2_LANG_PHI2_MAX_TOKENS'))
        if os.getenv('MV2_LANG_MISTRAL_MAX_TOKENS'):
            self.language.mistral_max_tokens = int(os.getenv('MV2_LANG_MISTRAL_MAX_TOKENS'))
        
        # Resource settings
        if os.getenv('MV2_GPU_MEMORY_BUDGET'):
            self.resources.gpu_memory_budget_gb = float(os.getenv('MV2_GPU_MEMORY_BUDGET'))
        if os.getenv('MV2_CPU_MEMORY_BUDGET'):
            self.resources.cpu_memory_budget_gb = float(os.getenv('MV2_CPU_MEMORY_BUDGET'))
        
        # Processing settings
        if os.getenv('MV2_DEFAULT_MODE'):
            self.processing.default_mode = os.getenv('MV2_DEFAULT_MODE')
        if os.getenv('MV2_ENABLE_CACHING'):
            self.processing.enable_caching = os.getenv('MV2_ENABLE_CACHING').lower() == 'true'
        
        # Server settings
        if os.getenv('MV2_SERVER_PORT'):
            self.server.port = int(os.getenv('MV2_SERVER_PORT'))
        if os.getenv('MV2_SERVER_DEBUG'):
            self.server.debug = os.getenv('MV2_SERVER_DEBUG').lower() == 'true'
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config_data = {
                'vision': asdict(self.vision),
                'language': asdict(self.language),
                'resources': asdict(self.resources),
                'processing': asdict(self.processing),
                'server': asdict(self.server)
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"💾 Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"❌ Error saving config: {e}")
    
    def get_model_config(self, model_type: str, tier: Optional[str] = None) -> Dict[str, Any]:
        """Get model-specific configuration"""
        if model_type == "blip":
            return {
                "model_name": self.vision.blip_model,
                "image_size": self.vision.image_size,
                "use_fp16": self.vision.use_fp16,
                "gpu_resident": self.vision.gpu_resident
            }
        elif model_type == "clip":
            return {
                "model_name": self.vision.clip_model,
                "image_size": self.vision.image_size,
                "use_fp16": self.vision.use_fp16,
                "gpu_resident": self.vision.gpu_resident
            }
        elif model_type == "yolo":
            return {
                "model_name": self.vision.yolo_model,
                "image_size": self.vision.image_size,
                "gpu_resident": False  # CPU-only as per architecture
            }
        elif model_type == "flan":
            return {
                "model_name": self.language.flan_model,
                "max_tokens": self.language.flan_max_tokens,
                "gpu_resident": self.language.flan_gpu_resident
            }
        elif model_type == "phi2":
            return {
                "model_name": self.language.phi2_model,
                "max_tokens": self.language.phi2_max_tokens,
                "quantization": self.language.phi2_quantization,
                "mixed_precision": self.language.phi2_mixed_precision
            }
        elif model_type == "mistral":
            return {
                "model_name": self.language.mistral_model,
                "max_tokens": self.language.mistral_max_tokens,
                "cpu_offload": self.language.mistral_cpu_offload,
                "offload_ratio": self.language.mistral_offload_ratio
            }
        else:
            logger.warning(f"⚠️ Unknown model type: {model_type}")
            return {}
    
    def get_processing_config(self, mode: str) -> Dict[str, Any]:
        """Get processing configuration for specific mode"""
        base_config = {
            "enable_caching": self.processing.enable_caching,
            "cache_ttl": self.processing.cache_ttl_minutes,
            "timeout": self.processing.timeout_seconds
        }
        
        if mode == "fast":
            base_config.update({
                "models": ["blip", "flan"],
                "max_tokens": self.language.flan_max_tokens,  # Now 50 for better responses
                "include_clip": False,
                "include_objects": False
            })
        elif mode == "balanced":
            base_config.update({
                "models": ["blip", "clip", "phi2"],
                "max_tokens": self.language.phi2_max_tokens,
                "include_clip": True,
                "include_objects": False
            })
        elif mode == "rich":
            base_config.update({
                "models": ["blip", "clip", "mistral"],
                "max_tokens": self.language.mistral_max_tokens,
                "include_clip": True,
                "include_objects": True  # Full analysis
            })
        
        return base_config
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []
        
        # Check GPU memory budget
        if self.resources.gpu_memory_budget_gb < 3.0:
            issues.append("GPU memory budget too low for concrete architecture (minimum 3GB)")
        
        # Check CPU memory for Mistral offloading
        if self.resources.cpu_memory_budget_gb < 8.0 and self.language.mistral_cpu_offload:
            issues.append("CPU memory too low for Mistral-7B offloading (minimum 8GB)")
        
        # Check token limits
        if self.language.flan_max_tokens > 50:
            issues.append("Flan-T5 max tokens too high for fast mode (recommended ≤20)")
        
        if self.language.phi2_max_tokens > 100:
            issues.append("Phi-2 max tokens too high for balanced mode (recommended ≤50)")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "config_summary": {
                "vision_models": ["BLIP", "CLIP", "YOLO"],
                "language_models": ["Flan-T5", "Phi-2", "Mistral-7B"],
                "gpu_budget_gb": self.resources.gpu_memory_budget_gb,
                "cpu_budget_gb": self.resources.cpu_memory_budget_gb,
                "default_mode": self.processing.default_mode
            }
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Get caching configuration"""
        cache_dir = Path("cache/monovision_v2")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            "enabled": self.processing.enable_caching,
            "cache_dir": str(cache_dir),
            "ttl_minutes": self.processing.cache_ttl_minutes,
            "max_size_mb": self.resources.cache_size_mb,
            "models_cache_dir": str(cache_dir / "models"),
            "results_cache_dir": str(cache_dir / "results"),
            "embeddings_cache_dir": str(cache_dir / "embeddings")
        }
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"""
MonoVision V2 Configuration:
├── Vision: {self.vision.blip_model}, {self.vision.clip_model}
├── Language: {self.language.flan_model}, {self.language.phi2_model}, {self.language.mistral_model}
├── Resources: {self.resources.gpu_memory_budget_gb}GB GPU, {self.resources.cpu_memory_budget_gb}GB CPU
├── Processing: {self.processing.default_mode} mode, caching={self.processing.enable_caching}
└── Server: {self.server.host}:{self.server.port}
"""

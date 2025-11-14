"""
Resource Manager for MonoVision V2
Handles GPU/CPU allocation according to the concrete architecture
"""

import torch
import psutil
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ResourceAllocation:
    """Resource allocation tracking"""
    gpu_memory_used: float
    gpu_memory_total: float
    cpu_memory_used: float
    cpu_memory_total: float
    gpu_utilization: float

class ResourceManager:
    """
    Manages resource allocation for the concrete tiered architecture:
    - GPU: BLIP (800MB) + CLIP (500MB) + Flan-T5/Phi-2 (800MB-1.5GB)
    - CPU: YOLOv8n + Mistral-7B offloaded layers + preprocessing
    - RAM: 16GB for caching and offloaded model layers
    """
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.device = "cuda" if self.gpu_available else "cpu"
        
        # Resource budgets (based on GTX 1650 4GB + 16GB RAM)
        self.gpu_memory_budget = 4.0 * 1024  # 4GB in MB
        self.gpu_memory_reserved = 800       # 800MB for CUDA overhead
        self.gpu_memory_available = self.gpu_memory_budget - self.gpu_memory_reserved
        
        # Memory allocation tracker
        self.allocated_models = {}
        
        logger.info(f"🎯 Resource Manager initialized - Device: {self.device}")
        if self.gpu_available:
            gpu_props = torch.cuda.get_device_properties(0)
            logger.info(f"📊 GPU: {gpu_props.name}, Memory: {gpu_props.total_memory / 1024**3:.1f}GB")
    
    async def allocate_gpu_memory(self):
        """Allocate GPU memory according to the concrete architecture"""
        if not self.gpu_available:
            logger.warning("⚠️ No GPU available, falling back to CPU-only mode")
            return
        
        logger.info("🎯 Allocating GPU memory for concrete architecture")
        
        # Expected allocation:
        expected_allocation = {
            "BLIP": 800,     # MB
            "CLIP": 500,     # MB
            "Flan-T5": 800,  # MB (Tier 1)
            "Buffer": 800    # MB (for dynamic loading)
        }
        
        total_expected = sum(expected_allocation.values())
        
        if total_expected > self.gpu_memory_available:
            logger.warning(f"⚠️ Expected allocation ({total_expected}MB) exceeds available ({self.gpu_memory_available}MB)")
        else:
            logger.info(f"✅ GPU allocation plan: {expected_allocation}")
            logger.info(f"📊 Total planned: {total_expected}MB / {self.gpu_memory_available}MB available")
    
    def register_model_allocation(self, model_name: str, memory_mb: float):
        """Register a model's memory allocation"""
        self.allocated_models[model_name] = memory_mb
        logger.info(f"📝 Registered {model_name}: {memory_mb}MB")
    
    def unregister_model_allocation(self, model_name: str):
        """Unregister a model's memory allocation"""
        if model_name in self.allocated_models:
            memory_freed = self.allocated_models.pop(model_name)
            logger.info(f"🗑️ Unregistered {model_name}: freed {memory_freed}MB")
    
    def get_gpu_usage(self) -> Dict[str, Any]:
        """Get current GPU usage statistics"""
        if not self.gpu_available:
            return {"available": False}
        
        try:
            # Get current GPU memory usage
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**2    # MB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            
            # Calculate utilization
            utilization = (memory_allocated / memory_total) * 100
            
            return {
                "available": True,
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_mb": round(memory_allocated, 1),
                "memory_reserved_mb": round(memory_reserved, 1),
                "memory_total_mb": round(memory_total, 1),
                "memory_free_mb": round(memory_total - memory_reserved, 1),
                "utilization_percent": round(utilization, 1),
                "allocated_models": self.allocated_models,
                "budget_status": {
                    "planned_usage_mb": sum(self.allocated_models.values()),
                    "budget_mb": self.gpu_memory_available,
                    "overhead_mb": self.gpu_memory_reserved,
                    "within_budget": sum(self.allocated_models.values()) <= self.gpu_memory_available
                }
            }
        except Exception as e:
            logger.error(f"❌ Error getting GPU usage: {e}")
            return {
                "available": False, 
                "device_name": "Unknown GPU", 
                "memory_allocated_mb": 0,
                "memory_total_mb": 0,
                "error": str(e)
            }
    
    def get_cpu_usage(self) -> Dict[str, Any]:
        """Get current CPU and RAM usage statistics"""
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_total_gb = memory.total / 1024**3
            memory_used_gb = memory.used / 1024**3
            memory_available_gb = memory.available / 1024**3
            memory_percent = memory.percent
            
            return {
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
                "memory_total_gb": round(memory_total_gb, 1),
                "memory_used_gb": round(memory_used_gb, 1),
                "memory_available_gb": round(memory_available_gb, 1),
                "memory_percent": memory_percent,
                "suitable_for_mistral": memory_available_gb >= 8.0  # Need 8GB for Mistral-7B offload
            }
        except Exception as e:
            logger.error(f"❌ Error getting CPU usage: {e}")
            return {"error": str(e)}
    
    def get_system_status(self) -> ResourceAllocation:
        """Get comprehensive system resource status"""
        gpu_info = self.get_gpu_usage()
        cpu_info = self.get_cpu_usage()
        
        return ResourceAllocation(
            gpu_memory_used=gpu_info.get("memory_allocated_mb", 0),
            gpu_memory_total=gpu_info.get("memory_total_mb", 0),
            cpu_memory_used=cpu_info.get("memory_used_gb", 0) * 1024,  # Convert to MB
            cpu_memory_total=cpu_info.get("memory_total_gb", 0) * 1024,  # Convert to MB
            gpu_utilization=gpu_info.get("utilization_percent", 0)
        )
    
    def can_load_model(self, model_name: str, estimated_memory_mb: float) -> bool:
        """Check if a model can be loaded given current allocations"""
        current_allocation = sum(self.allocated_models.values())
        total_after_loading = current_allocation + estimated_memory_mb
        
        can_load = total_after_loading <= self.gpu_memory_available
        
        logger.info(f"🤔 Can load {model_name} ({estimated_memory_mb}MB)? "
                   f"Current: {current_allocation}MB, After: {total_after_loading}MB, "
                   f"Budget: {self.gpu_memory_available}MB → {'✅ Yes' if can_load else '❌ No'}")
        
        return can_load
    
    def suggest_model_unloading(self, required_memory_mb: float) -> list:
        """Suggest which models to unload to free memory"""
        current_allocation = sum(self.allocated_models.values())
        memory_needed = required_memory_mb - (self.gpu_memory_available - current_allocation)
        
        if memory_needed <= 0:
            return []  # No unloading needed
        
        # Sort models by memory usage (largest first) - could be improved with usage patterns
        sorted_models = sorted(self.allocated_models.items(), key=lambda x: x[1], reverse=True)
        
        suggestions = []
        freed_memory = 0
        
        for model_name, memory_mb in sorted_models:
            if model_name in ["BLIP", "CLIP", "Flan-T5"]:  # Core models - don't suggest
                continue
            
            suggestions.append(model_name)
            freed_memory += memory_mb
            
            if freed_memory >= memory_needed:
                break
        
        logger.info(f"💡 To free {memory_needed}MB, suggest unloading: {suggestions}")
        return suggestions
    
    async def cleanup_gpu_memory(self):
        """Clean up GPU memory and run garbage collection"""
        if self.gpu_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("🧹 GPU memory cleaned up")
        
        # Also run Python garbage collection
        import gc
        gc.collect()
        logger.info("🧹 System garbage collection completed")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get comprehensive memory information (alias for get_gpu_usage for compatibility)"""
        try:
            gpu_info = self.get_gpu_usage()
            cpu_info = self.get_cpu_usage()
            
            return {
                "gpu": gpu_info,
                "cpu": cpu_info,
                "total_allocated_models": len(self.allocated_models),
                "gpu_budget_mb": self.gpu_memory_available,
                "allocated_models": dict(self.allocated_models)
            }
        except Exception as e:
            logger.error(f"❌ Error getting memory info: {e}")
            return {"error": str(e)}

"""Language processing modules for MonoVision"""

# Tier 1: Flan-T5-Small (Fast, 800MB)
from .tier1_flan import FlanT5Processor

# Tier 2: Phi-2 (Balanced, 1.5GB 4-bit) - FIXED VERSION
# from .tier2_phi import Phi2Processor

# Tier 3: API models (Rich, external)
from .tier3_api import ThirdPartyAPIProcessor

__all__ = [
    "FlanT5Processor",
    # "Phi2Processor",  # Uncomment when deploying with GPU
    "ThirdPartyAPIProcessor",
]

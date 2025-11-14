"""
Tier 3 Language Model: Third-Party API (Rich Mode)
Uses external APIs for comprehensive analysis instead of local models
"""

import asyncio
import logging
import time
import hashlib
import os
from typing import Dict, Any, Tuple, Optional
from enum import Enum

import aiohttp
import json
from ..core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)

class APIProvider(Enum):
    """Supported third-party API providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    TOGETHER = "together"
    HUGGINGFACE = "huggingface"
    GEMINI = "gemini"

class ThirdPartyAPIProcessor:
    """
    Tier 3 Language Processor using Third-Party APIs
    - Uses external APIs instead of local models
    - Supports multiple providers (OpenAI, Anthropic, Together AI, etc.)
    - Used in Rich mode: BLIP + CLIP + API
    - Target: 50-100 tokens, comprehensive analysis
    """

    def __init__(self, resource_manager: Optional[ResourceManager] = None):
        self.resource_manager = resource_manager or ResourceManager()

        # API Configuration - FASTER SETTINGS
        self.api_provider = APIProvider.GEMINI  # Default to Gemini Flash
        self.api_key = self._get_api_key()
        self.model_name = "gemini-1.5-flash"  # Fast Gemini Flash model
        self.max_tokens = 150  # Increased for richer responses
        self.timeout = 25  # REDUCED from 60s to 25s for faster responses
        self.retry_delay = 1  # Quick retry delay
        self.max_retries = 2  # Reduced retries for speed

        # API endpoints
        self.api_endpoints = {
            APIProvider.OPENAI: "https://api.openai.com/v1/chat/completions",
            APIProvider.ANTHROPIC: "https://api.anthropic.com/v1/messages",
            APIProvider.TOGETHER: "https://api.together.xyz/v1/chat/completions",
            APIProvider.HUGGINGFACE: "https://api-inference.huggingface.co/models/",
            APIProvider.GEMINI: "https://generativelanguage.googleapis.com/v1beta/models/"
        }

        # Performance tracking
        self.generation_count = 0
        self.total_generation_time = 0.0
        self.last_used = time.time()

        # CACHE FOR API RESPONSES
        self.response_cache = {}
        self.cache_ttl = 300  # 5 minutes cache for API responses

        logger.info(f"🎯 Third-Party API Processor (Tier 3) initialized - Provider: {self.api_provider.value}, Timeout: {self.timeout}s")

    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment or config"""

        # Load .env file if it exists
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass  # python-dotenv not installed, continue without it

        # Try different environment variable names
        api_keys = {
            APIProvider.OPENAI: ["OPENAI_API_KEY", "OPENAI_KEY"],
            APIProvider.ANTHROPIC: ["ANTHROPIC_API_KEY", "ANTHROPIC_KEY"],
            APIProvider.TOGETHER: ["TOGETHER_API_KEY", "TOGETHER_KEY"],
            APIProvider.HUGGINGFACE: ["HUGGINGFACE_API_KEY", "HF_API_KEY"],
            APIProvider.GEMINI: ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GEMINI_KEY"]
        }

        for env_var in api_keys.get(self.api_provider, []):
            api_key = os.getenv(env_var)
            if api_key:
                logger.info(f"✅ Found API key for {self.api_provider.value}")
                return api_key

        logger.warning(f"⚠️ No API key found for {self.api_provider.value}")
        return None

    def _get_cache_key(self, prompt: str, image_hash: str = None) -> str:
        """Generate cache key for API responses"""
        cache_content = f"{prompt}:{image_hash or 'no_image'}:{self.model_name}"
        return hashlib.md5(cache_content.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached API response if available and not expired"""
        if cache_key in self.response_cache:
            cached_data = self.response_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_ttl:
                logger.info(f"✅ API cache hit: {cache_key[:8]}...")
                return cached_data['response']
            else:
                # Remove expired cache
                del self.response_cache[cache_key]
        return None

    def _cache_response(self, cache_key: str, response: str):
        """Cache API response"""
        self.response_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }
        # Keep cache size manageable
        if len(self.response_cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(self.response_cache.keys(),
                               key=lambda k: self.response_cache[k]['timestamp'])[:20]
            for key in oldest_keys:
                del self.response_cache[key]

    async def initialize(self):
        """Initialize API processor (no heavy model loading needed)"""
        logger.info(f"🚀 Initializing {self.api_provider.value} API Processor (Tier 3: Rich Mode)")

        if not self.api_key:
            raise ValueError(f"API key required for {self.api_provider.value}")

        # Test API connection
        await self._test_api_connection()

        logger.info(f"✅ {self.api_provider.value} API initialized successfully")

    async def _test_api_connection(self):
        """Test API connection with a simple request"""
        try:
            test_prompt = "Hello, this is a test."

            if self.api_provider == APIProvider.OPENAI:
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": test_prompt}],
                    "max_tokens": 10
                }
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

            elif self.api_provider == APIProvider.ANTHROPIC:
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": test_prompt}],
                    "max_tokens": 10
                }
                headers = {
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }

            elif self.api_provider == APIProvider.TOGETHER:
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": test_prompt}],
                    "max_tokens": 10
                }
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

            elif self.api_provider == APIProvider.GEMINI:
                # Gemini API test
                payload = {
                    "contents": [{"parts": [{"text": test_prompt}]}],
                    "generationConfig": {
                        "temperature": 0.7,
                        "topK": 40,
                        "topP": 0.95,
                        "maxOutputTokens": 50,
                    }
                }
                headers = {"Content-Type": "application/json"}
                # Gemini uses API key as query parameter, not header

            else:  # HuggingFace
                payload = {"inputs": test_prompt, "parameters": {"max_new_tokens": 10}}
                headers = {"Authorization": f"Bearer {self.api_key}"}

            # Build URL with API key for Gemini
            if self.api_provider == APIProvider.GEMINI:
                url = f"{self.api_endpoints[self.api_provider]}{self.model_name}:generateContent?key={self.api_key}"
            else:
                url = self.api_endpoints[self.api_provider]

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info("✅ API connection test successful")
                    else:
                        logger.warning(f"⚠️ API test returned status {response.status}")

        except Exception as e:
            logger.warning(f"⚠️ API connection test failed: {e}")

    async def _call_openai_api(self, prompt: str, max_tokens: int) -> Tuple[str, int]:
        """Call OpenAI API with caching"""
        # Generate cache key
        cache_key = self._get_cache_key(prompt)

        # Check cache first
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response, len(cached_response.split()) * 1.2  # Estimate tokens

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are an expert image analyst providing comprehensive analysis."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": min(max_tokens, self.max_tokens),
            "temperature": 0.7,
            "top_p": 0.9
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_endpoints[APIProvider.OPENAI],
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data["choices"][0]["message"]["content"]

                    # Cache the successful response
                    self._cache_response(cache_key, response_text)

                    token_count = data["usage"]["completion_tokens"]
                    return response_text, token_count
                else:
                    error_data = await response.json()
                    raise Exception(f"OpenAI API error: {error_data}")

    async def _call_anthropic_api(self, prompt: str, max_tokens: int) -> Tuple[str, int]:
        """Call Anthropic API with caching"""
        # Generate cache key
        cache_key = self._get_cache_key(prompt)

        # Check cache first
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response, len(cached_response.split()) * 1.2  # Estimate tokens

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": min(max_tokens, self.max_tokens),
            "temperature": 0.7,
            "top_p": 0.9
        }

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_endpoints[APIProvider.ANTHROPIC],
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data["content"][0]["text"]

                    # Cache the successful response
                    self._cache_response(cache_key, response_text)

                    token_count = data["usage"]["output_tokens"]
                    return response_text, token_count
                else:
                    error_data = await response.json()
                    raise Exception(f"Anthropic API error: {error_data}")

    async def _call_together_api(self, prompt: str, max_tokens: int) -> Tuple[str, int]:
        """Call Together AI API with caching"""
        # Generate cache key
        cache_key = self._get_cache_key(prompt)

        # Check cache first
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response, len(cached_response.split()) * 1.2  # Estimate tokens

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are an expert image analyst providing comprehensive analysis."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": min(max_tokens, self.max_tokens),
            "temperature": 0.7,
            "top_p": 0.9
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_endpoints[APIProvider.TOGETHER],
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    response_text = data["choices"][0]["message"]["content"]

                    # Cache the successful response
                    self._cache_response(cache_key, response_text)

                    token_count = data["usage"]["completion_tokens"]
                    return response_text, token_count
                else:
                    error_data = await response.json()
                    raise Exception(f"Together AI API error: {error_data}")

    async def _call_huggingface_api(self, prompt: str, max_tokens: int) -> Tuple[str, int]:
        """Call HuggingFace Inference API with caching"""
        # Generate cache key
        cache_key = self._get_cache_key(prompt)

        # Check cache first
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response, len(cached_response.split()) * 1.2  # Estimate tokens

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": min(max_tokens, self.max_tokens),
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            }
        }

        headers = {"Authorization": f"Bearer {self.api_key}"}

        url = f"{self.api_endpoints[APIProvider.HUGGINGFACE]}{self.model_name}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, list) and data:
                        response_text = data[0].get("generated_text", "").replace(prompt, "").strip()
                    else:
                        response_text = data.get("generated_text", "").replace(prompt, "").strip()

                    # Cache the successful response
                    self._cache_response(cache_key, response_text)

                    # Estimate token count (rough approximation)
                    token_count = len(response_text.split()) * 1.3  # Rough token estimation
                    return response_text, int(token_count)
                else:
                    error_data = await response.json()
                    raise Exception(f"HuggingFace API error: {error_data}")

    async def _call_gemini_api(self, prompt: str, max_tokens: int, image_data: bytes = None, vision_results: Dict[str, Any] = None) -> Tuple[str, int]:
        """Call Google Gemini API with enhanced caching, faster timeouts, and improved error handling"""
        logger.info(f"🔄 Calling Gemini API with image_data: {image_data is not None}, prompt length: {len(prompt)}")

        # Generate cache key
        image_hash = vision_results.get('image_hash', '') if vision_results else ''
        cache_key = self._get_cache_key(prompt, image_hash)

        # Check cache first
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response, len(cached_response.split()) * 1.2  # Estimate tokens

        # Validate and compress image if provided
        if image_data:
            try:
                image_data, image_format = await self._prepare_image_for_gemini(image_data)
                logger.info(f"📷 Prepared image for Gemini: format={image_format}, size={len(image_data)} bytes")
            except Exception as e:
                logger.warning(f"⚠️ Failed to prepare image for Gemini: {e}")
                # Fall back to text-only mode
                image_data = None

        # Build the request payload
        contents = []

        if image_data:
            # Convert image to base64
            import base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')

            # Use detected format
            mime_type = f"image/{image_format}"

            logger.info(f"📷 Sending image to Gemini: format={image_format}, size={len(image_data)} bytes, b64_length={len(image_b64)}")

            contents.append({
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_b64
                        }
                    }
                ]
            })
        else:
            contents.append({
                "parts": [{"text": prompt}]
            })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": min(max_tokens, self.max_tokens),
            }
        }

        url = f"{self.api_endpoints[APIProvider.GEMINI]}{self.model_name}:generateContent?key={self.api_key}"

        logger.info(f"🌐 Gemini API URL: {url}")
        logger.info(f"📦 Payload structure: contents={len(contents)}, parts={len(contents[0]['parts'])}")

        # FASTER RETRY LOGIC with exponential backoff
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"⏱️ Starting Gemini API request (attempt {attempt + 1}/{self.max_retries + 1}) with {self.timeout}s timeout...")
                start_request_time = time.time()

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)  # FASTER TIMEOUT
                    ) as response:
                        request_time = time.time() - start_request_time
                        logger.info(f"📡 Gemini API response status: {response.status} (took {request_time:.2f}s)")

                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"✅ Gemini API response received: {bool(data.get('candidates'))}")

                            if "candidates" in data and data["candidates"]:
                                response_text = data["candidates"][0]["content"]["parts"][0]["text"]

                                # Cache the successful response
                                self._cache_response(cache_key, response_text)

                                # Estimate token count (rough approximation)
                                token_count = len(response_text.split()) * 1.2  # Gemini tokens are roughly 1.2x word count
                                return response_text, int(token_count)
                            else:
                                logger.error(f"❌ No candidates in Gemini response: {data}")
                                if attempt < self.max_retries:
                                    logger.info(f"🔄 Retrying Gemini API call (attempt {attempt + 2}/{self.max_retries + 1})...")
                                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                                    continue
                                raise Exception("No response generated by Gemini")
                        else:
                            error_data = await response.json()
                            logger.error(f"❌ Gemini API error response: {error_data}")

                            # Check for specific error types
                            if response.status == 429:  # Rate limit
                                if attempt < self.max_retries:
                                    logger.info(f"🔄 Rate limited, retrying Gemini API call (attempt {attempt + 2}/{self.max_retries + 1})...")
                                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Longer pause for rate limit
                                    continue
                            elif response.status >= 500:  # Server error
                                if attempt < self.max_retries:
                                    logger.info(f"🔄 Server error, retrying Gemini API call (attempt {attempt + 2}/{self.max_retries + 1})...")
                                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                                    continue

                            raise Exception(f"Gemini API error: {error_data}")

            except asyncio.TimeoutError:
                logger.error(f"⏰ Gemini API request timed out after {self.timeout}s (attempt {attempt + 1}/{self.max_retries + 1})")
                if attempt < self.max_retries:
                    logger.info(f"🔄 Timeout, retrying Gemini API call (attempt {attempt + 2}/{self.max_retries + 1})...")
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise Exception(f"Gemini API request timed out after {self.timeout} seconds on all {self.max_retries + 1} attempts. The image might be too large or the API is busy.")
            except aiohttp.ClientError as e:
                logger.error(f"🌐 Network error calling Gemini API: {e}")
                if attempt < self.max_retries:
                    logger.info(f"🔄 Network error, retrying Gemini API call (attempt {attempt + 2}/{self.max_retries + 1})...")
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise Exception(f"Network error calling Gemini API after {self.max_retries + 1} attempts: {str(e)}")
            except Exception as e:
                logger.error(f"❌ Unexpected error in Gemini API call: {e}")
                if attempt < self.max_retries:
                    logger.info(f"🔄 Unexpected error, retrying Gemini API call (attempt {attempt + 2}/{self.max_retries + 1})...")
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                raise

        # If we get here, all retries failed
        raise Exception(f"Gemini API call failed after {self.max_retries + 1} attempts")

    async def _prepare_image_for_gemini(self, image_data: bytes) -> Tuple[bytes, str]:
        """Prepare image for Gemini API by validating format, checking size, and compressing if needed"""
        try:
            from PIL import Image
            import io

            # Load image
            image = Image.open(io.BytesIO(image_data))

            # Detect format
            original_format = image.format.lower() if image.format else 'jpeg'

            # Validate format (Gemini supports JPEG, PNG, WebP, HEIC, HEIF)
            supported_formats = ['jpeg', 'jpg', 'png', 'webp', 'heic', 'heif']
            if original_format not in supported_formats:
                logger.info(f"🔄 Converting {original_format} to JPEG for Gemini compatibility")
                original_format = 'jpeg'

            # Check image size (Gemini has limits)
            max_dimension = 2048  # Gemini's recommended max dimension
            max_file_size = 20 * 1024 * 1024  # 20MB limit

            width, height = image.size
            needs_resize = width > max_dimension or height > max_dimension

            if needs_resize:
                logger.info(f"📏 Resizing image from {width}x{height} to fit Gemini limits")

                # Calculate new dimensions maintaining aspect ratio
                if width > height:
                    new_width = min(width, max_dimension)
                    new_height = int(height * (new_width / width))
                else:
                    new_height = min(height, max_dimension)
                    new_width = int(width * (new_height / height))

                # Resize image
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert to RGB if necessary (for JPEG)
            if original_format == 'jpeg' and image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')

            # Save to bytes with compression
            output_buffer = io.BytesIO()

            if original_format == 'jpeg':
                image.save(output_buffer, format='JPEG', quality=85, optimize=True)
            elif original_format == 'png':
                image.save(output_buffer, format='PNG', optimize=True)
            elif original_format == 'webp':
                image.save(output_buffer, format='WebP', quality=85)
            else:
                # Default to JPEG for unsupported formats
                image = image.convert('RGB')
                image.save(output_buffer, format='JPEG', quality=85, optimize=True)
                original_format = 'jpeg'

            compressed_data = output_buffer.getvalue()

            # Final size check
            if len(compressed_data) > max_file_size:
                logger.warning(f"⚠️ Compressed image still too large ({len(compressed_data)} bytes), further compression needed")

                # Try more aggressive compression
                if original_format == 'jpeg':
                    output_buffer = io.BytesIO()
                    image.save(output_buffer, format='JPEG', quality=70, optimize=True)
                    compressed_data = output_buffer.getvalue()

                if len(compressed_data) > max_file_size:
                    raise Exception(f"Image too large even after compression ({len(compressed_data)} bytes > {max_file_size} bytes)")

            logger.info(f"✅ Image prepared: {original_format}, {len(compressed_data)} bytes, {image.size}")
            return compressed_data, original_format

        except Exception as e:
            logger.error(f"❌ Failed to prepare image for Gemini: {e}")
            raise Exception(f"Image preparation failed: {str(e)}")

    async def generate_response(self, vision_results: Dict[str, Any], user_query: str, max_tokens: int = 150, context_prompt: Dict = None) -> Tuple[str, int]:
        """
        Generate ENHANCED comprehensive response using third-party API with caching

        Args:
            vision_results: Results from vision fusion layer
            user_query: User's question
            max_tokens: Maximum tokens to generate (increased for richer responses)

        Returns:
            Tuple of (response_text, token_count)
        """
        start_time = time.time()
        self.last_used = time.time()

        try:
            # Build enhanced prompt for rich mode
            prompt = self._build_rich_prompt(vision_results, user_query)

            # For Gemini, we can send the image directly if available
            image_data = vision_results.get('image_data') if isinstance(vision_results, dict) else None
            logger.info(f"🔍 Gemini API - vision_results type: {type(vision_results)}, has_image_data: {'image_data' in vision_results if isinstance(vision_results, dict) else False}, image_data_size: {len(image_data) if image_data else 0} bytes")

            # Generate response based on provider with caching
            if self.api_provider == APIProvider.GEMINI:
                response_text, token_count = await self._call_gemini_api(prompt, max_tokens, image_data, vision_results)
            elif self.api_provider == APIProvider.OPENAI:
                response_text, token_count = await self._call_openai_api(prompt, max_tokens)
            elif self.api_provider == APIProvider.ANTHROPIC:
                response_text, token_count = await self._call_anthropic_api(prompt, max_tokens)
            elif self.api_provider == APIProvider.TOGETHER:
                response_text, token_count = await self._call_together_api(prompt, max_tokens)
            elif self.api_provider == APIProvider.HUGGINGFACE:
                response_text, token_count = await self._call_huggingface_api(prompt, max_tokens)
            else:
                raise ValueError(f"Unsupported API provider: {self.api_provider}")

            # Enhanced response cleaning for rich mode
            response_text = self._clean_rich_response(response_text)

            # Update performance metrics
            generation_time = time.time() - start_time
            self.generation_count += 1
            self.total_generation_time += generation_time

            logger.info(f"🎯 {self.api_provider.value} API generated {token_count} tokens in {generation_time:.2f}s (cached: {generation_time < 0.1})")

            return response_text, token_count

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"❌ API generation error: {e}")
            logger.error(f"❌ Full traceback: {error_details}")

            # Provide helpful error message
            if "timeout" in str(e).lower():
                error_msg = "The analysis is taking longer than expected. This might be due to high server load or large image processing. Please try again in a moment."
            elif "rate limit" in str(e).lower():
                error_msg = "The API is currently rate-limited. Please wait a moment before trying again."
            else:
                error_msg = f"Unable to generate comprehensive analysis: {str(e)}"

            return error_msg, 0

    def _build_rich_prompt(self, vision_results: Dict[str, Any], user_query: str) -> str:
        """
        Build ENHANCED comprehensive prompt for rich mode analysis with better structure
        Include all available vision information with improved formatting
        """
        # Extract comprehensive vision context
        caption = vision_results.get("caption", "")
        clip_keywords = vision_results.get("clip_keywords", [])
        objects = vision_results.get("objects", [])

        # Build detailed context with better structure
        context_parts = []

        # Main description
        if caption:
            context_parts.append(f"📝 Primary Description: {caption}")

        # Visual keywords with categorization
        if clip_keywords:
            # Ensure clip_keywords are strings
            if isinstance(clip_keywords, list):
                keyword_strings = [str(k) for k in clip_keywords]

                # Categorize keywords
                style_keywords = [k for k in keyword_strings if any(word in k.lower() for word in ['style', 'art', 'painting', 'photo', 'image'])]
                color_keywords = [k for k in keyword_strings if any(word in k.lower() for word in ['color', 'bright', 'dark', 'red', 'blue', 'green'])]
                content_keywords = [k for k in keyword_strings if k not in style_keywords and k not in color_keywords]

                if style_keywords:
                    context_parts.append(f"🎨 Visual Style: {', '.join(style_keywords)}")
                if color_keywords:
                    context_parts.append(f"🌈 Color & Lighting: {', '.join(color_keywords)}")
                if content_keywords:
                    context_parts.append(f"📋 Content Elements: {', '.join(content_keywords)}")
            else:
                context_parts.append(f"🔍 Visual Keywords: {str(clip_keywords)}")

        # Enhanced object analysis
        if objects:
            # Handle objects properly - they might be dicts or strings
            if isinstance(objects, list) and objects:
                object_strings = []
                for obj in objects:
                    if isinstance(obj, dict):
                        # Extract name or other relevant field from dict
                        obj_name = obj.get('name', obj.get('class', str(obj)))
                        confidence = obj.get('confidence', '')
                        if confidence and isinstance(confidence, (int, float)):
                            if confidence > 0.8:
                                confidence_level = "high"
                            elif confidence > 0.6:
                                confidence_level = "medium"
                            else:
                                confidence_level = "low"
                            object_strings.append(f"{obj_name} ({confidence_level} confidence)")
                        else:
                            object_strings.append(str(obj_name))
                    else:
                        # Already a string
                        object_strings.append(str(obj))

                # Group objects by type for better analysis
                people = [obj for obj in object_strings if any(word in obj.lower() for word in ['person', 'man', 'woman', 'child', 'face'])]
                vehicles = [obj for obj in object_strings if any(word in obj.lower() for word in ['car', 'truck', 'bus', 'bike', 'motorcycle'])]
                animals = [obj for obj in object_strings if any(word in obj.lower() for word in ['dog', 'cat', 'bird', 'horse', 'animal'])]
                other_objects = [obj for obj in object_strings if obj not in people and obj not in vehicles and obj not in animals]

                if people:
                    context_parts.append(f"👥 People: {', '.join(people)}")
                if vehicles:
                    context_parts.append(f"🚗 Vehicles: {', '.join(vehicles)}")
                if animals:
                    context_parts.append(f"🐾 Animals: {', '.join(animals)}")
                if other_objects:
                    context_parts.append(f"📦 Objects: {', '.join(other_objects)}")
            else:
                context_parts.append(f"🔍 Detected Elements: {str(objects)}")

        context = "\n".join(context_parts)

        if user_query.strip():
            # User has specific question - provide comprehensive analysis with enhanced structure
            prompt = f"""You are an expert visual analyst and art critic. Based on the detailed image analysis below, provide a comprehensive, insightful response to the user's question.

📊 IMAGE ANALYSIS:
{context}

❓ USER QUESTION: {user_query}

🎯 ANALYSIS REQUIREMENTS:
• Provide detailed, analytical insights based on visual elements
• Consider composition, lighting, mood, and artistic qualities
• Draw connections between different visual elements
• Be thorough but focused on the most relevant aspects
• Include specific observations about colors, textures, and spatial relationships
• If relevant, discuss potential symbolism or contextual meaning

💡 RESPONSE STRUCTURE:
1. Direct answer to the question
2. Supporting visual evidence from the analysis
3. Additional insights and observations
4. Professional analysis conclusion

Please provide a rich, detailed analysis that demonstrates deep visual understanding."""
        else:
            # No specific question - provide comprehensive description with enhanced analysis
            prompt = f"""You are an expert visual analyst. Based on the detailed image analysis below, provide a comprehensive description and professional analysis.

📊 IMAGE ANALYSIS:
{context}

🎯 ANALYSIS REQUIREMENTS:
• Describe the image with rich, detailed language
• Analyze composition, color palette, and visual hierarchy
• Discuss mood, atmosphere, and emotional impact
• Identify key visual elements and their relationships
• Consider lighting, texture, and spatial organization
• Provide insights about style, technique, and artistic qualities

💡 RESPONSE STRUCTURE:
1. Overall impression and primary subject
2. Detailed description of visual elements
3. Analysis of composition and technique
4. Discussion of mood and atmosphere
5. Professional insights and observations

Please provide a comprehensive visual analysis that demonstrates expert-level understanding."""

        return prompt

    def _clean_rich_response(self, response_text: str) -> str:
        """Clean up the generated comprehensive response"""
        # Remove common artifacts
        response_text = response_text.strip()

        # Ensure proper paragraph structure for long responses
        if len(response_text) > 200:
            # Add paragraph breaks for readability
            sentences = response_text.split('. ')
            if len(sentences) > 3:
                # Group sentences into paragraphs
                paragraphs = []
                current_paragraph = []

                for i, sentence in enumerate(sentences):
                    current_paragraph.append(sentence)

                    # Create paragraph break every 2-3 sentences
                    if (i + 1) % 3 == 0 and i > 0:
                        paragraphs.append('. '.join(current_paragraph) + '.')
                        current_paragraph = []

                # Add remaining sentences
                if current_paragraph:
                    paragraphs.append('. '.join(current_paragraph))

                response_text = '\n\n'.join(paragraphs)

        # Ensure proper ending
        if response_text and not response_text.endswith(('.', '!', '?')):
            response_text += "."

        return response_text

    def recently_used(self, threshold_minutes: int = 5) -> bool:
        """Check if API was used recently (shorter threshold for API)"""
        minutes_since_last_use = (time.time() - self.last_used) / 60
        return minutes_since_last_use < threshold_minutes

    async def unload(self):
        """Unload API processor (no heavy resources to free)"""
        logger.info(f"🗑️ Unloading {self.api_provider.value} API processor")
        logger.info("✅ API processor unloaded successfully")

    def get_stats(self) -> Dict[str, Any]:
        """Get API processing statistics"""
        avg_generation_time = (
            self.total_generation_time / self.generation_count
            if self.generation_count > 0 else 0
        )

        return {
            "tier": "Tier 3 (Rich)",
            "provider": self.api_provider.value,
            "model_name": self.model_name,
            "generation_count": self.generation_count,
            "total_generation_time": self.total_generation_time,
            "average_generation_time": avg_generation_time,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "has_api_key": self.api_key is not None,
            "last_used": self.last_used,
            "recently_used": self.recently_used()
        }

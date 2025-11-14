"""
Enhanced Cache Manager V3 for MonoVision
Unified structured caching with image hashes, embeddings, TTL, and object detection
"""

import asyncio
import logging
import time
import json
import hashlib
import pickle
import base64
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
from collections import OrderedDict
from PIL import Image
import io

logger = logging.getLogger(__name__)

@dataclass
class StructuredCacheEntry:
    """Structured cache entry for MonoVision V3"""
    image_hash: str
    image_size: Tuple[int, int]
    image_format: str
    thumbnail_base64: str
    caption: str
    keywords: List[str]
    objects: List[str]
    embeddings: List[float]
    ai_response: str
    processing_mode: str
    token_count: int
    processing_time: float
    timestamp: float
    ttl_expires: float
    access_count: int
    last_accessed: float

@dataclass
class ImageCacheInfo:
    """Image cache information"""
    hash: str
    size: Tuple[int, int]
    format: str
    thumbnail_base64: str
@dataclass
class CacheEntry:
    """Cache entry structure"""
    key: str
    data: Any
    timestamp: float
    access_count: int
    last_accessed: float
    size_bytes: int

class LRUCache:
    """LRU (Least Recently Used) cache implementation"""
    
    def __init__(self, max_size_mb: float = 512):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)  # Convert MB to bytes
        self.cache = OrderedDict()
        self.current_size = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache:
            entry = self.cache[key]
            entry.access_count += 1
            entry.last_accessed = time.time()
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return entry.data
        return None
    
    def put(self, key: str, data: Any) -> bool:
        """Put item in cache, return True if successful"""
        try:
            # Estimate size
            size_bytes = len(pickle.dumps(data))
            
            # Remove old entry if exists
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_size -= old_entry.size_bytes
                del self.cache[key]
            
            # Check if we need to evict items
            while self.current_size + size_bytes > self.max_size_bytes and self.cache:
                # Remove least recently used item
                oldest_key, oldest_entry = self.cache.popitem(last=False)
                self.current_size -= oldest_entry.size_bytes
                logger.debug(f"🗑️ Evicted cache entry: {oldest_key}")
            
            # Add new entry
            if self.current_size + size_bytes <= self.max_size_bytes:
                entry = CacheEntry(
                    key=key,
                    data=data,
                    timestamp=time.time(),
                    access_count=1,
                    last_accessed=time.time(),
                    size_bytes=size_bytes
                )
                self.cache[key] = entry
                self.current_size += size_bytes
                return True
            else:
                logger.warning(f"⚠️ Cache entry too large: {size_bytes} bytes")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error adding to cache: {e}")
            return False
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_accesses = sum(entry.access_count for entry in self.cache.values())
        
        return {
            "entries": len(self.cache),
            "size_mb": self.current_size / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "utilization_percent": (self.current_size / self.max_size_bytes) * 100,
            "total_accesses": total_accesses
        }

class EnhancedCacheManager:
    """
    Enhanced Cache Manager V3 for MonoVision
    - Unified structured caching with image hashes
    - TTL-based expiration  
    - Image thumbnails and metadata
    - Object detection results caching
    - Smart cache hits for similar images
    """
    
    def __init__(self, cache_dir: str = "cache/monovision_v3"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Short-term RAM caches (from original)
        self.results_cache = LRUCache(max_size_mb=256)
        self.embeddings_cache = LRUCache(max_size_mb=128) 
        self.images_cache = LRUCache(max_size_mb=128)
        
        # Enhanced cache settings
        self.ttl_hours = 24  # Default TTL: 24 hours
        self.similarity_threshold = 0.95  # Embedding similarity threshold
        self.thumbnail_size = (150, 150)  # Thumbnail dimensions
        
        # Long-term storage
        self.db_path = self.cache_dir / "monovision_v3_cache.db"
        self.db_conn = None
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        self.similarity_hits = 0
        
        logger.info("💾 Enhanced Cache Manager V3 initialized")
    """LRU (Least Recently Used) cache implementation"""
    
    def __init__(self, max_size_mb: float = 512):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)  # Convert MB to bytes
        self.cache = OrderedDict()
        self.current_size = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache:
            entry = self.cache[key]
            entry.access_count += 1
            entry.last_accessed = time.time()
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return entry.data
        return None
    
    def put(self, key: str, data: Any) -> bool:
        """Put item in cache, return True if successful"""
        try:
            # Estimate size
            size_bytes = len(pickle.dumps(data))
            
            # Remove old entry if exists
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_size -= old_entry.size_bytes
                del self.cache[key]
            
            # Check if we need to evict items
            while self.current_size + size_bytes > self.max_size_bytes and self.cache:
                # Remove least recently used item
                oldest_key, oldest_entry = self.cache.popitem(last=False)
                self.current_size -= oldest_entry.size_bytes
                logger.debug(f"🗑️ Evicted cache entry: {oldest_key}")
            
            # Add new entry
            if self.current_size + size_bytes <= self.max_size_bytes:
                entry = CacheEntry(
                    key=key,
                    data=data,
                    timestamp=time.time(),
                    access_count=1,
                    last_accessed=time.time(),
                    size_bytes=size_bytes
                )
                self.cache[key] = entry
                self.current_size += size_bytes
                return True
            else:
                logger.warning(f"⚠️ Cache entry too large: {size_bytes} bytes")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error adding to cache: {e}")
            return False
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_accesses = sum(entry.access_count for entry in self.cache.values())
        
        return {
            "entries": len(self.cache),
            "size_mb": self.current_size / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "utilization_percent": (self.current_size / self.max_size_bytes) * 100,
            "total_accesses": total_accesses
        }

class CacheManager:
    """
    Comprehensive cache management for MonoVision V2
    - Short-term: RAM cache for recent results and embeddings
    - Long-term: SQLite database for persistent storage
    """
    
    def __init__(self, cache_dir: str = "cache/monovision_v2"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Short-term RAM caches
        self.results_cache = LRUCache(max_size_mb=256)  # Processing results
        self.embeddings_cache = LRUCache(max_size_mb=128)  # CLIP embeddings
        self.images_cache = LRUCache(max_size_mb=128)  # Image data
        
        # Long-term storage
        self.db_path = self.cache_dir / "monovision_cache.db"
        self.db_conn = None
        
        # Cache settings
        self.ttl_minutes = 60  # Time to live for results
        self.max_db_entries = 10000  # Maximum database entries
        
        # Performance tracking
        self.hit_count = 0
        self.miss_count = 0
        
        logger.info("💾 Cache Manager initialized")
    
    async def initialize(self):
        """Initialize cache manager"""
        try:
            # Initialize SQLite database
            self._init_database()
            logger.info("✅ Cache Manager initialized successfully")
        except Exception as e:
            logger.error(f"❌ Error initializing cache: {e}")
            # Set db_conn to None to prevent further errors
            self.db_conn = None
    
    def _init_database(self):
        """Initialize SQLite database for long-term storage"""
        try:
            self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            
            # Create tables
            self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS results_cache (
                cache_key TEXT PRIMARY KEY,
                data BLOB,
                timestamp REAL,
                access_count INTEGER DEFAULT 1,
                last_accessed REAL,
                metadata TEXT
            )
            """)
            
            self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings_cache (
                image_hash TEXT PRIMARY KEY,
                embedding BLOB,
                keywords TEXT,
                timestamp REAL,
                last_accessed REAL
            )
            """)
            
            self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS session_memory (
                session_id TEXT,
                image_hash TEXT,
                query TEXT,
                response TEXT,
                timestamp REAL,
                PRIMARY KEY (session_id, image_hash)
            )
            """)
            
            # Create indexes
            self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON results_cache(timestamp)")
            self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON results_cache(last_accessed)")
            self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON session_memory(session_id)")
            
            self.db_conn.commit()
            
            logger.info("📄 Database initialized")
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            self.db_conn = None
    
    def _ensure_db(self):
        """Ensure database connection exists and is working"""
        if self.db_conn is None:
            try:
                self._init_database()
            except Exception as e:
                logger.error(f"❌ Failed to initialize database: {e}")
                self.db_conn = None
        
        # Test connection
        if self.db_conn:
            try:
                self.db_conn.execute("SELECT 1")
                return True
            except Exception as e:
                logger.error(f"❌ Database connection test failed: {e}")
                self.db_conn = None
                return False
        return False
    
    def generate_cache_key(self, image_data: bytes, query: str, mode: str) -> str:
        """Generate unique cache key for a request"""
        # Create hash from image data
        image_hash = hashlib.md5(image_data).hexdigest()[:16]
        
        # Create hash from query and mode
        query_hash = hashlib.md5(f"{query}:{mode}".encode()).hexdigest()[:8]
        
        return f"{image_hash}_{query_hash}"
    
    async def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if available and not expired"""
        try:
            # Check RAM cache first (fastest)
            result = self.results_cache.get(cache_key)
            if result:
                # Check if not expired
                if time.time() - result.get("timestamp", 0) < self.ttl_minutes * 60:
                    self.hit_count += 1
                    logger.debug(f"✅ RAM cache hit: {cache_key}")
                    return result
                else:
                    logger.debug(f"⏰ RAM cache expired: {cache_key}")
            
            # Check database cache only if connection is available
            if not self._ensure_db():
                self.miss_count += 1
                return None
            
            cursor = self.db_conn.execute(
                "SELECT data, timestamp, access_count FROM results_cache WHERE cache_key = ?",
                (cache_key,)
            )
            row = cursor.fetchone()
            
            if row:
                data_blob, timestamp, access_count = row
                
                # Check if not expired
                if time.time() - timestamp < self.ttl_minutes * 60:
                    # Deserialize data
                    result = pickle.loads(data_blob)
                    
                    # Update access count and last accessed
                    self.db_conn.execute(
                        "UPDATE results_cache SET access_count = ?, last_accessed = ? WHERE cache_key = ?",
                        (access_count + 1, time.time(), cache_key)
                    )
                    self.db_conn.commit()
                    
                    # Add back to RAM cache
                    self.results_cache.put(cache_key, result)
                    
                    self.hit_count += 1
                    logger.debug(f"✅ DB cache hit: {cache_key}")
                    return result
                else:
                    # Remove expired entry
                    self.db_conn.execute("DELETE FROM results_cache WHERE cache_key = ?", (cache_key,))
                    self.db_conn.commit()
                    logger.debug(f"⏰ DB cache expired and removed: {cache_key}")
            
            self.miss_count += 1
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting cached result: {e}")
            self.miss_count += 1
            return None
    
    async def cache_result(self, cache_key: str, result: Any):
        """Cache a processing result"""
        try:
            # Add timestamp
            if hasattr(result, '__dict__'):
                result_dict = result.__dict__.copy()
            else:
                result_dict = result.copy() if isinstance(result, dict) else {"data": result}
            
            result_with_timestamp = result_dict
            result_with_timestamp["timestamp"] = time.time()
            result_with_timestamp["cached"] = True
            
            # Add to RAM cache
            self.results_cache.put(cache_key, result_with_timestamp)
            
            # Add to database cache only if connection is available
            if self._ensure_db():
                data_blob = pickle.dumps(result_with_timestamp)
                metadata = json.dumps({
                    "mode": getattr(result, 'mode', 'unknown').value if hasattr(getattr(result, 'mode', None), 'value') else str(getattr(result, 'mode', 'unknown')),
                    "processing_time": getattr(result, 'processing_time', 0)
                })
                
                self.db_conn.execute("""
                INSERT OR REPLACE INTO results_cache 
                (cache_key, data, timestamp, last_accessed, metadata)
                VALUES (?, ?, ?, ?, ?)
                """, (cache_key, data_blob, time.time(), time.time(), metadata))
                
                self.db_conn.commit()
                
                logger.debug(f"💾 Cached result: {cache_key}")
                
                # Clean up old entries if needed
                await self._cleanup_old_entries()
            else:
                logger.debug(f"💾 RAM cached only (no DB): {cache_key}")
            
        except Exception as e:
            logger.error(f"❌ Error caching result: {e}")
    
    async def cache_embedding(self, image_hash: str, embedding: List[float], keywords: List[str]):
        """Cache CLIP embedding and keywords with enhanced structure"""
        try:
            # Add to RAM cache
            embedding_data = {
                "embedding": embedding,
                "keywords": keywords,
                "timestamp": time.time(),
                "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "access_count": 1
            }
            self.embeddings_cache.put(image_hash, embedding_data)
            
            # Add to database with enhanced structure
            if self._ensure_db():
                embedding_blob = pickle.dumps(embedding)
                keywords_json = json.dumps(keywords)
                
                self.db_conn.execute("""
                INSERT OR REPLACE INTO embeddings_cache 
                (image_hash, embedding, keywords, timestamp, last_accessed)
                VALUES (?, ?, ?, ?, ?)
                """, (image_hash, embedding_blob, keywords_json, time.time(), time.time()))
                
                self.db_conn.commit()
                
                logger.debug(f"🔗 Cached embedding: {image_hash}")
            
        except Exception as e:
            logger.error(f"❌ Error caching embedding: {e}")
    
    async def cache_vision_result(self, image_hash: str, caption: str, keywords: List[str], objects: List[str] = None):
        """Cache comprehensive vision analysis results"""
        try:
            vision_data = {
                "caption": caption,
                "keywords": keywords,
                "objects": objects or [],
                "timestamp": time.time(),
                "cached_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "access_count": 1
            }
            
            # Cache in RAM
            cache_key = f"vision_{image_hash}"
            self.results_cache.put(cache_key, vision_data)
            
            # Cache in database
            if self._ensure_db():
                # Create vision_cache table if not exists
                self.db_conn.execute("""
                CREATE TABLE IF NOT EXISTS vision_cache (
                    image_hash TEXT PRIMARY KEY,
                    caption TEXT,
                    keywords TEXT,
                    objects TEXT,
                    timestamp REAL,
                    last_accessed REAL,
                    access_count INTEGER DEFAULT 1
                )
                """)
                
                self.db_conn.execute("""
                INSERT OR REPLACE INTO vision_cache 
                (image_hash, caption, keywords, objects, timestamp, last_accessed, access_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    image_hash, 
                    caption, 
                    json.dumps(keywords), 
                    json.dumps(objects or []),
                    time.time(), 
                    time.time(), 
                    1
                ))
                
                self.db_conn.commit()
                logger.debug(f"🔍 Cached vision result: {image_hash}")
                
        except Exception as e:
            logger.error(f"❌ Error caching vision result: {e}")
    
    async def get_cached_vision_result(self, image_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached vision analysis results"""
        try:
            # Check RAM cache first
            cache_key = f"vision_{image_hash}"
            result = self.results_cache.get(cache_key)
            if result:
                return result
            
            # Check database
            if not self._ensure_db():
                return None
                
            cursor = self.db_conn.execute(
                "SELECT caption, keywords, objects, timestamp, access_count FROM vision_cache WHERE image_hash = ?",
                (image_hash,)
            )
            row = cursor.fetchone()
            
            if row:
                caption, keywords_json, objects_json, timestamp, access_count = row
                
                # Check if not too old (vision results can be kept longer)
                if time.time() - timestamp < 24 * 60 * 60:  # 24 hours TTL
                    result = {
                        "caption": caption,
                        "keywords": json.loads(keywords_json) if keywords_json else [],
                        "objects": json.loads(objects_json) if objects_json else [],
                        "timestamp": timestamp,
                        "access_count": access_count,
                        "cached": True
                    }
                    
                    # Update access count
                    self.db_conn.execute(
                        "UPDATE vision_cache SET access_count = ?, last_accessed = ? WHERE image_hash = ?",
                        (access_count + 1, time.time(), image_hash)
                    )
                    self.db_conn.commit()
                    
                    # Add back to RAM cache
                    self.results_cache.put(cache_key, result)
                    
                    logger.debug(f"✅ Vision cache hit: {image_hash}")
                    return result
                else:
                    # Remove expired entry
                    self.db_conn.execute("DELETE FROM vision_cache WHERE image_hash = ?", (image_hash,))
                    self.db_conn.commit()
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting cached vision result: {e}")
            return None
    
    async def get_cached_embedding(self, image_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached CLIP embedding"""
        try:
            # Check RAM cache first
            result = self.embeddings_cache.get(image_hash)
            if result:
                return result
            
            # Check database
            cursor = self.db_conn.execute(
                "SELECT embedding, keywords, timestamp FROM embeddings_cache WHERE image_hash = ?",
                (image_hash,)
            )
            row = cursor.fetchone()
            
            if row:
                embedding_blob, keywords_json, timestamp = row
                
                # Check if not too old (embeddings can be kept longer)
                if time.time() - timestamp < 24 * 60 * 60:  # 24 hours
                    embedding = pickle.loads(embedding_blob)
                    keywords = json.loads(keywords_json)
                    
                    result = {
                        "embedding": embedding,
                        "keywords": keywords,
                        "timestamp": timestamp
                    }
                    
                    # Add back to RAM cache
                    self.embeddings_cache.put(image_hash, result)
                    
                    return result
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting cached embedding: {e}")
            return None
    
    async def store_session_memory(self, session_id: str, image_hash: str, query: str, response: str):
        """Store session memory for multi-turn conversation"""
        try:
            self.db_conn.execute("""
            INSERT OR REPLACE INTO session_memory 
            (session_id, image_hash, query, response, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """, (session_id, image_hash, query, response, time.time()))
            
            self.db_conn.commit()
            
        except Exception as e:
            logger.error(f"❌ Error storing session memory: {e}")
    
    async def get_session_memory(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent session memory"""
        try:
            cursor = self.db_conn.execute("""
            SELECT image_hash, query, response, timestamp 
            FROM session_memory 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
            """, (session_id, limit))
            
            rows = cursor.fetchall()
            
            return [
                {
                    "image_hash": row[0],
                    "query": row[1],
                    "response": row[2],
                    "timestamp": row[3]
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"❌ Error getting session memory: {e}")
            return []
    
    async def _cleanup_old_entries(self):
        """Clean up old cache entries"""
        try:
            # Remove expired results
            cutoff_time = time.time() - (self.ttl_minutes * 60)
            self.db_conn.execute("DELETE FROM results_cache WHERE timestamp < ?", (cutoff_time,))
            
            # Limit total entries (keep most recently accessed)
            self.db_conn.execute("""
            DELETE FROM results_cache 
            WHERE cache_key NOT IN (
                SELECT cache_key FROM results_cache 
                ORDER BY last_accessed DESC 
                LIMIT ?
            )
            """, (self.max_db_entries,))
            
            # Remove old embeddings (older than 7 days)
            old_embedding_cutoff = time.time() - (7 * 24 * 60 * 60)
            self.db_conn.execute("DELETE FROM embeddings_cache WHERE timestamp < ?", (old_embedding_cutoff,))
            
            # Remove old session memory (older than 30 days)
            old_session_cutoff = time.time() - (30 * 24 * 60 * 60)
            self.db_conn.execute("DELETE FROM session_memory WHERE timestamp < ?", (old_session_cutoff,))
            
            self.db_conn.commit()
            
        except Exception as e:
            logger.error(f"❌ Error cleaning up cache: {e}")
    
    def clear_cache(self):
        """Clear all caches"""
        try:
            # Clear RAM caches
            self.results_cache.clear()
            self.embeddings_cache.clear()
            self.images_cache.clear()
            
            # Clear database
            self.db_conn.execute("DELETE FROM results_cache")
            self.db_conn.execute("DELETE FROM embeddings_cache")
            self.db_conn.execute("DELETE FROM session_memory")
            self.db_conn.commit()
            
            logger.info("🧹 All caches cleared")
            
        except Exception as e:
            logger.error(f"❌ Error clearing cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            # Initialize database counts
            db_results_count = 0
            db_embeddings_count = 0
            db_sessions_count = 0
            
            # Get database stats only if connection exists
            if self.db_conn:
                cursor = self.db_conn.execute("SELECT COUNT(*) FROM results_cache")
                db_results_count = cursor.fetchone()[0]
                
                cursor = self.db_conn.execute("SELECT COUNT(*) FROM embeddings_cache")
                db_embeddings_count = cursor.fetchone()[0]
                
                cursor = self.db_conn.execute("SELECT COUNT(*) FROM session_memory")
                db_sessions_count = cursor.fetchone()[0]
            
            # Calculate hit rate
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "ram_cache": {
                    "results": self.results_cache.get_stats(),
                    "embeddings": self.embeddings_cache.get_stats(),
                    "images": self.images_cache.get_stats()
                },
                "database": {
                    "results_count": db_results_count,
                    "embeddings_count": db_embeddings_count,
                    "sessions_count": db_sessions_count,
                    "db_size_mb": self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0,
                    "connected": self.db_conn is not None
                },
                "performance": {
                    "hit_count": self.hit_count,
                    "miss_count": self.miss_count,
                    "hit_rate_percent": hit_rate
                },
                "settings": {
                    "ttl_minutes": self.ttl_minutes,
                    "max_db_entries": self.max_db_entries
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting cache stats: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Clean up cache manager"""
        try:
            if self.db_conn:
                self.db_conn.close()
            logger.info("✅ Cache Manager cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during cache cleanup: {e}")

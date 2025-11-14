"""
Enhanced Cache Manager V3 for MonoVision
Structured caching with image hashes, embeddings, and TTL
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
    caption: str
    keywords: List[str]
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

class EnhancedCacheManager:
    """
    Enhanced Cache Manager V3 for MonoVision
    - Structured caching with image hashes
    - TTL-based expiration
    - Image thumbnails and metadata
    - Smart cache hits for similar images
    """
    
    def __init__(self, cache_dir: str = "cache/monovision_v3"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache settings
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
    
    async def initialize(self):
        """Initialize enhanced cache manager"""
        try:
            self._init_database()
            await self._cleanup_expired_entries()
            logger.info("✅ Enhanced Cache Manager V3 ready")
        except Exception as e:
            logger.error(f"❌ Error initializing enhanced cache: {e}")
            self.db_conn = None
    
    def _init_database(self):
        """Initialize SQLite database with enhanced schema"""
        try:
            self.db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
            
            # Create enhanced tables
            self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS structured_cache (
                image_hash TEXT PRIMARY KEY,
                image_width INTEGER,
                image_height INTEGER,
                image_format TEXT,
                thumbnail_base64 TEXT,
                caption TEXT,
                keywords TEXT,
                embeddings BLOB,
                ai_response TEXT,
                processing_mode TEXT,
                token_count INTEGER,
                processing_time REAL,
                timestamp REAL,
                ttl_expires REAL,
                access_count INTEGER DEFAULT 1,
                last_accessed REAL
            )
            """)
            
            self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS similarity_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_hash TEXT,
                embedding_snippet BLOB,
                timestamp REAL,
                FOREIGN KEY (image_hash) REFERENCES structured_cache (image_hash)
            )
            """)
            
            self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_stats (
                date TEXT PRIMARY KEY,
                total_requests INTEGER,
                cache_hits INTEGER,
                similarity_hits INTEGER,
                total_processing_time REAL
            )
            """)
            
            # Create indexes for performance
            self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_ttl_expires ON structured_cache(ttl_expires)")
            self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON structured_cache(last_accessed)")
            self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON structured_cache(timestamp)")
            self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_processing_mode ON structured_cache(processing_mode)")
            
            self.db_conn.commit()
            logger.info("📄 Enhanced database schema initialized")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            self.db_conn = None
    
    def generate_image_hash(self, image_data: bytes) -> str:
        """Generate SHA-256 hash for image data"""
        return hashlib.sha256(image_data).hexdigest()
    
    def create_thumbnail(self, image_data: bytes) -> str:
        """Create base64 encoded thumbnail"""
        try:
            image = Image.open(io.BytesIO(image_data))
            image.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            thumbnail_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return thumbnail_base64
        except Exception as e:
            logger.error(f"❌ Error creating thumbnail: {e}")
            return ""
    
    def get_image_info(self, image_data: bytes) -> ImageCacheInfo:
        """Extract image information"""
        try:
            image = Image.open(io.BytesIO(image_data))
            return ImageCacheInfo(
                hash=self.generate_image_hash(image_data),
                size=(image.width, image.height),
                format=image.format or "UNKNOWN",
                thumbnail_base64=self.create_thumbnail(image_data)
            )
        except Exception as e:
            logger.error(f"❌ Error getting image info: {e}")
            return ImageCacheInfo("", (0, 0), "ERROR", "")
    
    async def get_cached_result(self, image_data: bytes, query: str, mode: str) -> Optional[StructuredCacheEntry]:
        """Get cached result for image + query combination"""
        if not self.db_conn:
            return None
        
        try:
            image_hash = self.generate_image_hash(image_data)
            current_time = time.time()
            
            # Check exact match first
            cursor = self.db_conn.execute("""
            SELECT * FROM structured_cache 
            WHERE image_hash = ? AND ttl_expires > ?
            """, (image_hash, current_time))
            
            row = cursor.fetchone()
            
            if row:
                # Update access statistics
                self.db_conn.execute("""
                UPDATE structured_cache 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE image_hash = ?
                """, (current_time, image_hash))
                self.db_conn.commit()
                
                # Convert row to StructuredCacheEntry
                entry = self._row_to_cache_entry(row)
                self.hit_count += 1
                
                logger.info(f"✅ Cache hit for image {image_hash[:8]}...")
                return entry
            
            # If no exact match, check for similar images (optional feature)
            # This could be implemented with embedding similarity
            
            self.miss_count += 1
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting cached result: {e}")
            return None
    
    async def cache_result(
        self, 
        image_data: bytes, 
        caption: str, 
        keywords: List[str],
        embeddings: List[float],
        ai_response: str,
        processing_mode: str,
        token_count: int,
        processing_time: float,
        ttl_hours: Optional[int] = None
    ) -> bool:
        """Cache a complete processing result"""
        if not self.db_conn:
            return False
        
        try:
            # Get image information
            image_info = self.get_image_info(image_data)
            
            # Calculate TTL
            current_time = time.time()
            ttl_expires = current_time + (ttl_hours or self.ttl_hours) * 3600
            
            # Store in database
            embeddings_blob = pickle.dumps(embeddings)
            keywords_json = json.dumps(keywords)
            
            self.db_conn.execute("""
            INSERT OR REPLACE INTO structured_cache (
                image_hash, image_width, image_height, image_format, thumbnail_base64,
                caption, keywords, embeddings, ai_response, processing_mode,
                token_count, processing_time, timestamp, ttl_expires, 
                access_count, last_accessed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                image_info.hash, image_info.size[0], image_info.size[1], image_info.format,
                image_info.thumbnail_base64, caption, keywords_json, embeddings_blob,
                ai_response, processing_mode, token_count, processing_time,
                current_time, ttl_expires, 1, current_time
            ))
            
            self.db_conn.commit()
            
            logger.info(f"💾 Cached result for image {image_info.hash[:8]}... (TTL: {ttl_hours or self.ttl_hours}h)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error caching result: {e}")
            return False
    
    def _row_to_cache_entry(self, row) -> StructuredCacheEntry:
        """Convert database row to StructuredCacheEntry"""
        return StructuredCacheEntry(
            image_hash=row[0],
            image_size=(row[1], row[2]),
            caption=row[5],
            keywords=json.loads(row[6]) if row[6] else [],
            embeddings=pickle.loads(row[7]) if row[7] else [],
            ai_response=row[8],
            processing_mode=row[9],
            token_count=row[10],
            processing_time=row[11],
            timestamp=row[12],
            ttl_expires=row[13],
            access_count=row[14],
            last_accessed=row[15]
        )
    
    async def get_image_thumbnail(self, image_hash: str) -> Optional[str]:
        """Get base64 thumbnail for an image hash"""
        if not self.db_conn:
            return None
        
        try:
            cursor = self.db_conn.execute(
                "SELECT thumbnail_base64 FROM structured_cache WHERE image_hash = ?",
                (image_hash,)
            )
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"❌ Error getting thumbnail: {e}")
            return None
    
    async def _cleanup_expired_entries(self):
        """Clean up expired cache entries"""
        if not self.db_conn:
            return
        
        try:
            current_time = time.time()
            
            # Remove expired entries
            cursor = self.db_conn.execute(
                "DELETE FROM structured_cache WHERE ttl_expires < ?",
                (current_time,)
            )
            deleted_count = cursor.rowcount
            
            # Clean up orphaned similarity index entries
            self.db_conn.execute("""
            DELETE FROM similarity_index 
            WHERE image_hash NOT IN (SELECT image_hash FROM structured_cache)
            """)
            
            self.db_conn.commit()
            
            if deleted_count > 0:
                logger.info(f"🧹 Cleaned up {deleted_count} expired cache entries")
                
        except Exception as e:
            logger.error(f"❌ Error cleaning up cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            stats = {
                "performance": {
                    "hit_count": self.hit_count,
                    "miss_count": self.miss_count,
                    "similarity_hits": self.similarity_hits,
                    "hit_rate_percent": (self.hit_count / (self.hit_count + self.miss_count) * 100) if (self.hit_count + self.miss_count) > 0 else 0
                },
                "storage": {
                    "total_entries": 0,
                    "db_size_mb": 0,
                    "expired_entries": 0
                },
                "settings": {
                    "ttl_hours": self.ttl_hours,
                    "similarity_threshold": self.similarity_threshold,
                    "connected": self.db_conn is not None
                }
            }
            
            if self.db_conn:
                # Get total entries
                cursor = self.db_conn.execute("SELECT COUNT(*) FROM structured_cache")
                stats["storage"]["total_entries"] = cursor.fetchone()[0]
                
                # Get expired entries count
                current_time = time.time()
                cursor = self.db_conn.execute(
                    "SELECT COUNT(*) FROM structured_cache WHERE ttl_expires < ?",
                    (current_time,)
                )
                stats["storage"]["expired_entries"] = cursor.fetchone()[0]
                
                # Get database size
                if self.db_path.exists():
                    stats["storage"]["db_size_mb"] = self.db_path.stat().st_size / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting cache stats: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Clean up cache manager"""
        try:
            await self._cleanup_expired_entries()
            if self.db_conn:
                self.db_conn.close()
            logger.info("✅ Enhanced Cache Manager cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during enhanced cache cleanup: {e}")

"""Redis cache service for risk map calculations."""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import redis
from app.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class RiskCache:
    """Redis cache for risk map calculations."""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize the risk cache.

        Args:
            redis_client: Optional Redis client (creates new if not provided)
        """
        self.redis_client = redis_client
        self._cache_enabled = True

        if self.redis_client is None:
            try:
                self.redis_client = redis.from_url(
                    settings.redis_url, decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Redis cache initialization failed: {str(e)}")
                logger.warning("Continuing without cache (graceful degradation)")
                self._cache_enabled = False
                self.redis_client = None

    def _generate_cache_key(
        self,
        start_time: datetime,
        end_time: datetime,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        grid_size_m: Optional[float] = None,
        use_hex: Optional[bool] = None,
    ) -> str:
        """
        Generate a cache key for risk map parameters.

        Args:
            start_time: Start time of the time window
            end_time: End time of the time window
            bbox: Optional bounding box
            grid_size_m: Optional grid size in meters
            use_hex: Optional hex grid flag

        Returns:
            Cache key string
        """
        # Create a hash of the parameters
        key_parts = [
            start_time.isoformat(),
            end_time.isoformat(),
        ]

        if bbox:
            key_parts.append(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")

        if grid_size_m:
            key_parts.append(f"grid_size:{grid_size_m}")

        if use_hex is not None:
            key_parts.append(f"hex:{use_hex}")

        key_string = ":".join(key_parts)
        key_hash = hashlib.md5(key_string.encode()).hexdigest()

        return f"risk_map:{key_hash}"

    def get_cached_risk_map(
        self,
        start_time: datetime,
        end_time: datetime,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        grid_size_m: Optional[float] = None,
        use_hex: Optional[bool] = None,
    ) -> Optional[Dict]:
        """
        Get cached risk map data.

        Args:
            start_time: Start time of the time window
            end_time: End time of the time window
            bbox: Optional bounding box
            grid_size_m: Optional grid size in meters
            use_hex: Optional hex grid flag

        Returns:
            Cached risk map data or None if not found
        """
        if not self._cache_enabled or self.redis_client is None:
            return None

        try:
            cache_key = self._generate_cache_key(
                start_time, end_time, bbox, grid_size_m, use_hex
            )
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                logger.debug(f"Cache hit for key: {cache_key}")
                return json.loads(cached_data)
            else:
                logger.debug(f"Cache miss for key: {cache_key}")
                return None

        except Exception as e:
            logger.error(f"Error getting cached risk map: {str(e)}")
            return None

    def set_cached_risk_map(
        self,
        risk_data: Dict,
        start_time: datetime,
        end_time: datetime,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        grid_size_m: Optional[float] = None,
        use_hex: Optional[bool] = None,
        ttl_seconds: Optional[int] = None,
    ):
        """
        Cache risk map data.

        Args:
            risk_data: Risk map data to cache
            start_time: Start time of the time window
            end_time: End time of the time window
            bbox: Optional bounding box
            grid_size_m: Optional grid size in meters
            use_hex: Optional hex grid flag
            ttl_seconds: Optional TTL in seconds (defaults to config value)
        """
        if not self._cache_enabled or self.redis_client is None:
            return

        try:
            cache_key = self._generate_cache_key(
                start_time, end_time, bbox, grid_size_m, use_hex
            )

            ttl = ttl_seconds or getattr(
                settings, "risk_cache_ttl_seconds", 3600
            )

            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(risk_data, default=str),
            )

            logger.debug(f"Cached risk map with key: {cache_key} (TTL: {ttl}s)")

        except Exception as e:
            logger.error(f"Error caching risk map: {str(e)}")

    def invalidate_cache(
        self, bbox: Optional[Tuple[float, float, float, float]] = None
    ):
        """
        Invalidate cache entries.

        Args:
            bbox: Optional bounding box to invalidate specific entries
                    If None, invalidates all risk_map entries
        """
        if not self._cache_enabled or self.redis_client is None:
            return

        try:
            if bbox is None:
                # Invalidate all risk_map entries
                pattern = "risk_map:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    logger.info(f"Invalidated {len(keys)} cache entries")
            else:
                # Invalidate entries matching bbox
                # This is approximate - we invalidate all entries since
                # we can't easily match bbox without recalculating keys
                pattern = "risk_map:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                    logger.info(
                        f"Invalidated {len(keys)} cache entries (bbox: {bbox})"
                    )

        except Exception as e:
            logger.error(f"Error invalidating cache: {str(e)}")

    def is_enabled(self) -> bool:
        """
        Check if cache is enabled and available.

        Returns:
            True if cache is enabled and available
        """
        return self._cache_enabled and self.redis_client is not None


# Singleton instance
_cache: Optional[RiskCache] = None


def get_risk_cache() -> RiskCache:
    """
    Get the singleton risk cache instance.

    Returns:
        RiskCache instance
    """
    global _cache
    if _cache is None:
        _cache = RiskCache()
    return _cache





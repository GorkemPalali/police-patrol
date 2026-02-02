import redis
from app.core.config import get_settings

settings = get_settings()

redis_client = redis.from_url(
    settings.redis_url,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_timeout=5,
)


def get_redis():
    """Dependency for getting Redis client"""
    return redis_client






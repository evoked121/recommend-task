import os
from typing import Optional

import redis.asyncio as aioredis
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB   = int(os.getenv("REDIS_DB", 0))

_redis_client: Optional[aioredis.Redis] = None

async def get_redis() -> aioredis.Redis:
    global _redis_client
    if not _redis_client:
        _redis_client = aioredis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            encoding="utf-8",
            decode_responses=True
        )
    return _redis_client

async def cache_user_prompt(user_id: str, prompt_text: str):
    r = await get_redis()
    key = f"prompt:{user_id}"
    await r.set(key, prompt_text)

async def get_user_prompt(user_id: str) -> Optional[str]:
    r = await get_redis()
    key = f"prompt:{user_id}"
    return await r.get(key)
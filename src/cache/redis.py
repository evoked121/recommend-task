import os
from typing import Optional, List
import json
from src.dataclass import Story

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

async def cache_story_embeddings(story_id: int, embedding: List[float]):
    r = await get_redis()
    key = f"story_embed:{story_id}"
    await r.set(key, json.dumps(embedding))

async def get_story_embedding(story_id: int) -> Optional[List[float]]:
    r = await get_redis()
    key = f"story_embed:{story_id}"
    data = await r.get(key)
    return json.loads(data) if data else None

async def cache_story_pool(stories: List[Story]):
    r = await get_redis()
    await r.set("story_pool", json.dumps(stories))

async def get_story_pool() -> Optional[List[Story]]:
    r = await get_redis()
    data = await r.get("story_pool")
    return json.loads(data) if data else None

async def get_story_embeddings_batch(story_ids: List[int]) -> List[Optional[List[float]]]:
    r = await get_redis()
    pipeline = r.pipeline()
    for story_id in story_ids:
        key = f"story_embed:{story_id}"
        pipeline.get(key)
    
    results = await pipeline.execute()
    return [json.loads(data) if data else None for data in results]
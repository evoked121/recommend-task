import os
import re
import json
from typing import List, Optional
from src.dataclass import Story
from dotenv import load_dotenv
from src.ai_agents.open_ai import OpenAiAgent
from src.cache.redis import get_story_embeddings_batch

open_ai_agent = OpenAiAgent()

load_dotenv()

async def generate_embedding(text: str) -> List[float]:
    resp = open_ai_agent.client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return resp.data[0].embedding

def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
    norm1 = sum(a * a for a in embedding1) ** 0.5
    norm2 = sum(b * b for b in embedding2) ** 0.5
    return dot_product / (norm1 * norm2)

async def prefilter_stories_with_embeddings(
    user_tags: List[str],
    story_pool: List[Story],
    top_k: int = 60
) -> List[Story]:
    user_embedding = await generate_embedding(" ".join(user_tags))
    
    story_ids = [s['id'] for s in story_pool]
    story_embeddings = await get_story_embeddings_batch(story_ids)
    
    similarities = []
    for story, story_embedding in zip(story_pool, story_embeddings):
        if story_embedding:
            similarity = calculate_similarity(user_embedding, story_embedding)
            similarities.append((story, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in similarities[:top_k]]

async def recommend_stories(
    prompt: str,
    user_tags: List[str],
    story_pool: List[Story]
) -> List[int]:
    filtered_stories = await prefilter_stories_with_embeddings(user_tags, story_pool)
    
    system_prompt = (
        "You are a lightning-fast recommendation engine. "
        "you must return exactly 10 story IDs—no more, no fewer—as a JSON array. "
        "Even if fewer than 10 stories strongly match, still provide 10 unique IDs by including the best possible alternatives. "
        "Do NOT include any additional commentary—only output the JSON array."
    )

    tags_str = json.dumps(user_tags, ensure_ascii=False)

    stories_text = ""
    for s in filtered_stories:
        stories_text += (
            f"ID: {int(s['id'])}\n"
            f"Title: {s['title']}\n"
            f"Intro: {s['intro']}\n"
            f"Tags: {', '.join(s['tags'])}\n\n"
        )

    user_prompt = (
        f"Prompt Instructions:\n{prompt}\n\n"
        f"User Tags: {tags_str}\n\n"
        f"Stories:\n{stories_text}"
    )

    resp = open_ai_agent.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=500
    )

    generated = resp.choices[0].message.content
    match = re.search(r"\[.*\]", generated, re.S)
    content = json.loads(match.group(0))
    return content
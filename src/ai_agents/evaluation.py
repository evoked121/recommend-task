from typing import Tuple, List, Dict
from src.ai_agents.recommend import recommend_stories
from src.dataclass import Story
import re
import os
import json
from src.ai_agents.open_ai import OpenAiAgent
from src.cache.redis import get_story_embeddings_batch

open_ai_agent = OpenAiAgent()

def simulate_user_tags(user_profile: List[str]) -> List[str]:
    system_prompt = (
        "You are a tag prediction assistant. Given a list of available preference tags for a user, "
        "select between 5 and 10 tags that best represent what this user would choose on Sekai’s first screen. "
        "Return EXACTLY a JSON array of strings (e.g., [\"tag1\", \"tag2\", ...]). "
        "Do NOT include any extra commentary, explanation, or Python/JSON syntax—only the JSON array itself."
    )
    user_prompt = f"Available Tags:\n{json.dumps(user_profile, ensure_ascii=False)}"

    resp = open_ai_agent.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=200
    )
    generated = resp.choices[0].message.content
    match = re.search(r"\[.*\]", generated, re.S)
    simulated_tags = json.loads(match.group(0))
    return simulated_tags

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

async def ground_truth_top10(user_profile: dict, story_pool: List[Story]) -> List[int]:
    filtered_stories = await prefilter_stories_with_embeddings(user_profile, story_pool)
    system_prompt = (
        "You are an expert story recommender. Given a user’s full profile and a list of Sekai stories "
        "(each has ID, title, tags, and intro)"
        "you must consider the entire set of tags together to determine which 10 stories best match the user’s interests. "
        "Do NOT include any additional commentary."
        "you must return exactly 10 story IDs—no more, no fewer—as a JSON array. "
        "Even if fewer than 10 stories strongly match, still provide 10 unique IDs by including the best possible alternatives. "
        "Under no circumstances should you return fewer or more than 10 IDs. "
    )

    stories_text = ""
    for s in filtered_stories:
        stories_text += str(s) + "\n"

    user_prompt = (
        f"User Profile:\n{json.dumps(user_profile, ensure_ascii=False)}\n\n"
        f"Stories:\n{stories_text}"
    )

    resp = open_ai_agent.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=200
    )
    generated = resp.choices[0].message.content
    match = re.search(r"\[.*\]", generated, re.S)
    content = json.loads(match.group(0))
    return content

async def evaluate_for_user(
    user_profile: List[str],
    prompt: str,
    story_pool: List[Story]
) -> Tuple[float, Dict]:
    user_tags = simulate_user_tags(user_profile)
    print(f"*****user_tags: {user_tags}")

    rec_ids = await recommend_stories(prompt, user_tags, story_pool)
    print(f"*****recomend id: {rec_ids}")

    gt_ids = await ground_truth_top10(user_profile, story_pool)
    print(f"*****truth id: {gt_ids}")
    true_positives = len(set(rec_ids) & set(gt_ids))
    precision = true_positives / 10.0

    failure_detail = {
        "simulated_tags": user_tags,
        "rec_ids": rec_ids,
        "gt_ids": gt_ids
    }

    return precision, failure_detail

import openai
from typing import Tuple, List, Dict
from recommend import recommend_stories
from dataclass import Story
import re
import os
import json


openai.api_key = os.getenv("OPENAI_API_KEY")

def parse_tags(text: str) -> List[str]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list) and all(isinstance(t, str) for t in data):
            return [t.strip() for t in data]
    except json.JSONDecodeError:
        pass
    return [t.strip() for t in text.split(",") if t.strip()]

def parse_ids(text: str) -> List[int]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list) and all(isinstance(n, int) for n in data):
            return data
    except json.JSONDecodeError:
        pass
    parts = [p for p in re.split(r"[\s,;\n]+", text) if p.strip().isdigit()]
    return [int(p) for p in parts]

async def simulate_user_tags(user_profile: dict) -> List[str]:
    system_prompt = (
        "You are a tag prediction assistant. Given a full user profile, "
        "output a JSON array of story tags that this user would select on Sekai’s first screen. "
        "Do NOT include any additional commentary—return exactly a JSON array of strings."
    )
    user_prompt = f"User Profile:\n{json.dumps(user_profile, ensure_ascii=False)}"

    resp = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=50
    )
    return parse_tags(resp.choices[0].message["content"])

async def ground_truth_top10(user_profile: dict, story_pool: List[Story]) -> List[int]:
    system_prompt = (
        "You are an expert story recommender. Given a user’s full profile and a list of 100 Sekai stories "
        "(each has ID, title, tags, and intro), return EXACTLY a JSON array of the 10 story IDs "
        "that best match this user’s preferences. Do NOT include any additional commentary."
    )

    stories_text = ""
    for s in story_pool:
        stories_text += str(s) + "\n"

    user_prompt = (
        f"User Profile:\n{json.dumps(user_profile, ensure_ascii=False)}\n\n"
        f"Stories:\n{stories_text}"
    )

    resp = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=100
    )
    return parse_ids(resp.choices[0].message["content"])

async def evaluate_for_user(
    user_profile: dict,
    prompt: str,
    story_pool: List[Story]
) -> Tuple[float, Dict]:
    user_tags = await simulate_user_tags(user_profile)

    rec_ids = await recommend_stories(prompt, user_tags, story_pool)

    gt_ids = await ground_truth_top10(user_profile, story_pool)
    true_positives = len(set(rec_ids) & set(gt_ids))
    precision = true_positives / 10.0

    failure_detail = {
        "user": user_profile.get("id"),
        "simulated_tags": user_tags,
        "rec_ids": rec_ids,
        "gt_ids": gt_ids
    }

    return precision, failure_detail

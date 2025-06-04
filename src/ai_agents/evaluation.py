from typing import Tuple, List, Dict
from src.ai_agents.recommend import recommend_stories
from src.dataclass import Story
import re
import os
import json
from src.ai_agents.open_ai import OpenAiAgent

open_ai_agent = OpenAiAgent()

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

def simulate_user_tags(user_profile: List[str]) -> List[str]:
    system_prompt = (
        "You are a tag prediction assistant. Given a list of available preference tags for a user, "
        "select between 5 and 8 tags that best represent what this user would choose on Sekai’s first screen. "
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

def ground_truth_top10(user_profile: dict, story_pool: List[Story]) -> List[int]:
    system_prompt = (
        "You are an expert story recommender. Given a user’s full profile and a list of Sekai stories "
        "(each has ID, title, tags, and intro), return EXACTLY a JSON array of the 10 story IDs "
        "that best match this user’s preferences. Do NOT include any additional commentary."
        "you must return exactly 10 story IDs—no more, no fewer—as a JSON array. "
        "Under no circumstances should you return fewer or more than 10 IDs. "
    )

    stories_text = ""
    for s in story_pool:
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
    return parse_ids(resp.choices[0].message.content)

async def evaluate_for_user(
    user_profile: List[str],
    prompt: str,
    story_pool: List[Story]
) -> Tuple[float, Dict]:
    user_tags = simulate_user_tags(user_profile)
    print(f"*****user_tags: {user_tags}")

    rec_ids = recommend_stories(prompt, user_tags, story_pool)
    print(f"*****recomend id: {rec_ids}")

    gt_ids = ground_truth_top10(user_profile, story_pool)
    print(f"*****truth id: {gt_ids}")
    true_positives = len(set(rec_ids) & set(gt_ids))
    precision = true_positives / 10.0

    failure_detail = {
        "simulated_tags": user_tags,
        "rec_ids": rec_ids,
        "gt_ids": gt_ids
    }

    return precision, failure_detail

import os
import re
import json
import openai
from typing import List
from dataclass import Story
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


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


async def recommend_stories(
    prompt: str,
    user_tags: List[str],
    story_pool: List[Story]
) -> List[int]:
    system_prompt = (
        "You are a lightning-fast recommendation engine. "
        "Given a set of user tags and a pool of stories (each with ID, title, intro, and tags), "
        "you must return EXACTLY a JSON array of the 10 story IDs that best match the user's tags. "
        "Do NOT include any additional commentary—only output the JSON array."
    )

    tags_str = json.dumps(user_tags, ensure_ascii=False)

    stories_text = ""
    for s in story_pool:
        stories_text += (
            f"ID: {s['id']}\n"
            f"Title: {s['title']}\n"
            f"Intro: {s['intro']}\n"
            f"Tags: {', '.join(s['tags'])}\n\n"
        )

    user_prompt = (
        f"Prompt Instructions:\n{prompt}\n\n"
        f"User Tags: {tags_str}\n\n"
        f"Stories:\n{stories_text}"
    )

    resp = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=100
    )

    content = resp.choices[0].message["content"]
    return parse_ids(content)


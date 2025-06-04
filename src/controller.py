# controller.py

import os
import json
import re
import asyncio
from typing import List, Dict, Any

import openai
from dotenv import load_dotenv

from src.ai_agents.prompt_optimizer import optimize_prompt
from src.ai_agents.evaluation import evaluate_for_user
from src.ai_agents.recommend import recommend_stories

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from typing import TypedDict
from dataclass import Story


seed_stories: List[Story] = [
    {
        "id": 217107,
        "title": "Stranger Who Fell From The Sky",
        "intro": "You are Devin, plummeting towards Orario with no memory of how you got here...",
        "tags": ["danmachi", "reincarnation", "heroic aspirations", "mystery origin", "teamwork", "loyalty", "protectiveness"]
    },
    {
        "id": 273613,
        "title": "Trapped Between Four Anime Legends!",
        "intro": "You're caught in a dimensional rift with four anime icons. Goku wants to spar...",
        "tags": ["crossover", "jujutsu kaisen", "dragon ball", "naruto", "isekai", "dimensional travel", "reverse harem"]
    },
    {
        "id": 235701,
        "title": "New Transfer Students vs. Class 1-A Bully",
        "intro": "You and Zeroku watch in disgust as Bakugo torments Izuku again...",
        "tags": ["my hero academia", "challenging authority", "bullying", "underdog", "disruptors"]
    },
    {
        "id": 214527,
        "title": "Zenitsu Touched Your Sister's WHAT?!",
        "intro": "Your peaceful afternoon at the Butterfly Estate shatters when Zenitsu accidentally gropes Nezuko...",
        "tags": ["demon slayer", "protective instincts", "comedic panic", "violent reactions"]
    },
    {
        "id": 263242,
        "title": "Principal's Daughter Dating Contest",
        "intro": "You are Yuji Itadori, facing off against Tanjiro and Naruto for Ochako's heart...",
        "tags": ["crossover", "romantic comedy", "forced proximity", "harem", "dating competition"]
    }
]

def load_test_users(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


async def expand_story_pool(seeds: List[Story], target_count: int = 100) -> List[Story]:
    seeds_text = json.dumps(seeds, ensure_ascii=False)
    system_prompt = (
        "You are a story generator assistant. Given a small list of Sekai-style stories "
        "(each with id, title, intro, tags), expand this list so that the total number "
        f"of stories is {target_count}. Return EXACTLY a JSON array of story objects, "
        "where each object has the fields: id (integer), title (string), intro (string), "
        "and tags (array of strings). Ensure new IDs start higher than existing seed IDs."
    )
    user_prompt = f"Seed Stories:\n{seeds_text}"

    resp = await openai.ChatCompletion.acreate(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=3000
    )
    generated = resp.choices[0].message["content"]

    all_stories: List[Story] = []
    try:
        data = json.loads(generated)
        if isinstance(data, list):
            all_stories = data
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", generated, re.S)
        if match:
            try:
                all_stories = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    if not all_stories or len(all_stories) < target_count:
        print("Warning: failed to parse expanded stories correctly. Using fallback duplication.")
        all_stories = seeds.copy()
        next_id = max(s["id"] for s in seeds) + 1
        while len(all_stories) < target_count:
            template = seeds[(len(all_stories) - len(seeds)) % len(seeds)]
            clone = {
                "id": next_id,
                "title": template["title"] + " (Clone)",
                "intro": template["intro"],
                "tags": template["tags"]
            }
            all_stories.append(clone)
            next_id += 1

    return all_stories[:target_count]


async def main():
    print("Expanding seed stories to ~100 with GPT-4...")
    story_pool = await expand_story_pool(seed_stories, target_count=100)
    print(f"Expansion complete: {len(story_pool)} stories loaded.\n")

    test_users = load_test_users("test_users.json")

    last_prompt = "Given user tags {tags}, return 10 story IDs from the pool."
    prev_avg = 0.0

    for it in range(1, 11):  # 上限 10 轮
        print(f"=== Iteration {it} ===")
        print(f"Using prompt:\n{last_prompt}\n")

        scores = []
        failures: List[Dict[str, Any]] = []

        user = test_users[0]
        score, detail = await evaluate_for_user(user, last_prompt, story_pool)
        print(f"User {detail['user']} → Precision@10 = {score:.2f}")
        scores.append(score)
        failures.append(detail)

        avg_score = scores[0]
        print(f"\nIteration {it} Precision@10 = {avg_score:.4f}\n")

        if it > 1 and (avg_score - prev_avg) < 0.005:
            print(f"Precision plateaued (Δ={avg_score - prev_avg:.4f}); stopping.\n")
            break

        print("Calling Prompt-Optimizer to generate new prompt...\n")
        new_prompt = await optimize_prompt(last_prompt, avg_score, failures)
        last_prompt = new_prompt
        prev_avg = avg_score

    print("Final prompt:")
    print(last_prompt)
    print("Loop finished.")


if __name__ == "__main__":
    asyncio.run(main())

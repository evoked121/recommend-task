import json
import re
import time
import asyncio
from typing import List, Dict, Any
from user import users

from dotenv import load_dotenv

from src.ai_agents.prompt_optimizer import optimize_prompt
from src.ai_agents.evaluation import evaluate_for_user
from src.dataclass import Story
from src.cache.redis import get_user_prompt, cache_user_prompt
from src.ai_agents.open_ai import OpenAiAgent

load_dotenv()

open_ai_agent = OpenAiAgent()

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


def expand_story_pool(seeds: List[Story], target_count: int = 100) -> List[Story]:
    seeds_text = json.dumps(seeds, ensure_ascii=False)
    system_prompt = (
        "You are a story generator assistant. Given a small list of Sekai-style stories "
        "(each with id, title, intro, tags), expand this list so that the total number "
        f"of stories is exactly {target_count}. Return EXACTLY a JSON array of story objects, "
        "where each object has the fields: id (integer), title (string), intro (string), "
        "and tags (array of strings). For speed and brevity, generate each new story "
        "with an intro of no more than 10 English words. Ensure new IDs start higher than existing seed IDs. "
        "Do NOT output any extra commentary—only the JSON array."
    )
    user_prompt = f"Seed Stories:\n{seeds_text}"

    resp = open_ai_agent.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=5000
    )
    generated = resp.choices[0].message.content
    match = re.search(r"\[.*\]", generated, re.S)
    stories = json.loads(match.group(0))
    return stories


async def main():
    print("Expanding seed stories to ~100 with GPT-4...")
    max_seconds= 10
    story_pool = expand_story_pool(seed_stories, target_count=30)

    test_users = users
    user = test_users[1]

    last_prompt = await get_user_prompt(f"prompt:user{user['id']}")
    last_prompt = last_prompt if last_prompt else "Given user tags, return 10 story IDs from the pool."

    scores = []
    start_time = time.time()
    iteration = 1

    while True:
        elapsed = time.time() - start_time
        if elapsed > max_seconds:
                await cache_user_prompt(f"prompt:user{user['id']}", last_prompt)
                print(f"[{user['id']}] Time limit reached ({elapsed:.1f}s). Stopping optimization.")
                break
        print(f"Using prompt:\n{last_prompt}\n")

        failures: List[Dict[str, Any]] = []
        
        score, detail = await evaluate_for_user(user['tags'], last_prompt, story_pool)
        print(f"Precision@10 = {score:.2f}")
        scores.append(score)
        failures.append(detail)

        print(f"\n Precision@10 = {score:.4f}\n")

        if score >= 0.8:
            print(f"Iteration:{iteration} Precision plateaued (Δ={score:.4f}); stopping.\n")
            break

        print("Calling Prompt-Optimizer to generate new prompt...\n")
        iteration += 1
        new_prompt = optimize_prompt(last_prompt, score, failures)
        print(new_prompt)
        last_prompt = new_prompt

    await cache_user_prompt(f"prompt:user{user['id']}", last_prompt)
    print("Final prompt:")
    print(scores)
    print(last_prompt)
    print("Loop finished.")


if __name__ == "__main__":
    asyncio.run(main())

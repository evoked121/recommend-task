import os
import json
import re
import ast
import asyncio
from typing import List, Dict, Any
from user import users

from dotenv import load_dotenv

from src.ai_agents.prompt_optimizer import optimize_prompt
from src.ai_agents.evaluation import evaluate_for_user, simulate_user_tags
from src.dataclass import Story
#from src.ai_agents.recommend import recommend_stories
from src.ai_agents.open_ai import OpenAiAgent

load_dotenv()

open_ai_agent = OpenAiAgent()

async def main():
    res = simulate_user_tags(user_profile=users[0])
    print(res)
    return


if __name__ == "__main__":
    asyncio.run(main())
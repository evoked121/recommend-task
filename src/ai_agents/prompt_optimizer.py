import os
from typing import List, Dict
from src.ai_agents.open_ai import OpenAiAgent

open_ai_agent = OpenAiAgent()

def optimize_prompt(
    last_prompt: str,
    last_score: float,
    failure_samples: List[Dict]
) -> str:
    
    system_prompt = (
        "You are a prompt optimization assistant whose job is to iteratively refine "
        "recommendation prompts so that a downstream recommendation agent can achieve "
        "higher Precision@10 for a group of users. "
        "You will be given:\n"
        "1) The prompt used in the last iteration.\n"
        "2) The last Precision@10 score achieved by that prompt.\n"
        "3) A list of failure examples, where each example includes:\n"
        "   - user: the user identifier\n"
        "   - rec_ids: the 10 story IDs recommended by the recommendation agent\n"
        "   - gt_ids: the 10 ground-truth story IDs for that user\n\n"
        "Your task: Analyze why the previous prompt underperformed (based on the average score "
        "and failure examples), and propose a new prompt that will guide the recommendation agent "
        "to select 10 story IDs more likely to match each user’s ground-truth preferences. "
        "Return ONLY the revised prompt string—do not include any explanation or commentary.\n"
    )

    user_content = (
        f"Last Prompt:\n{last_prompt}\n\n"
        f"Last Precision@10: {last_score:.4f}\n\n"
        f"Failure Examples:\n{failure_samples}\n\n"
        "Please produce a new prompt that will improve Precision@10.\n"
    )

    response = open_ai_agent.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0.3,
        max_tokens=200
    )

    new_prompt = response.choices[0].message.content.strip()
    return new_prompt


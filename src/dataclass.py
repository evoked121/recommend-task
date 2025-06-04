from typing import List
from dataclasses import dataclass

@dataclass
class Story:
    id: int
    title: str
    intro: str
    tags: List[str]

from json import load
from typing import TypedDict


class GameDict(TypedDict):
    name: str
    env: str
    slug: str
    description: str
    default_mode: str
    default_difficulty: str
    url: str


gameList: dict[str, GameDict]
with open("games.json", "r") as f:
    gameList = load(f)
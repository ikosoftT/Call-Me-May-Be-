
import json
from pathlib import Path
from typing import Dict, List


class VocabManager:
    def __init__(self, vocab_path: str) -> None:
        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}
        self._load_vocab(vocab_path)

    # My Private Method that Maps tokens
    def _load_vocab(self, vocab_path: str) -> None:
        path = Path(vocab_path)
        if not path.exists():
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON vocab file")
        # Checking HF vocab format is dict or list based index pos unique id
        if isinstance(data, dict):
            for token_id, token_str in data.items():
                self.token_to_id[token_str] = token_id
                self.id_to_token[token_id] = token_str
        elif isinstance(data, list):
            for token_id, token_str in enumerate(data):
                self.token_to_id[token_str] = token_id
                self.id_to_token[token_id] = token_str
        else:
            raise ValueError("Unsupported vocab file format.")

    # i'll give an id return its str token
    def get_token_str(self, token_id: int) -> str:
        if token_id not in self.id_to_token:
            raise KeyError(f"Token id {token_id} not found in vocab file.")
        return self.id_to_token[token_id]

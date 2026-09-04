# src/vocab.py

import json
from pathlib import Path
from typing import Dict, List


class VocabularyManager:
    """Manages mapping between token IDs and their string fragments

    using the model's vocabulary file provided by llm_sdk.
    """

    def __init__(self, vocab_path: str) -> None:
        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}
        self._load_vocab(vocab_path)

    def _load_vocab(self, vocab_path: str) -> None:
        path = Path(vocab_path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed vocabulary JSON: {e}") from e

        # Handle standard HuggingFace/tokenizer vocab formats (token -> id or id -> token)
        if isinstance(data, dict):
            for token_str, token_id in data.items():
                self.token_to_id[token_str] = int(token_id)
                self.id_to_token[int(token_id)] = token_str
        elif isinstance(data, list):
            for token_id, token_str in enumerate(data):
                self.token_to_id[token_str] = token_id
                self.id_to_token[token_id] = token_str
        else:
            raise ValueError("Unsupported vocabulary file format structure.")

    def get_token_string(self, token_id: int) -> str:
        """Retrieve the string representation for a specific token ID."""
        if token_id not in self.id_to_token:
            raise KeyError(f"Token ID {token_id} not found in vocabulary mapping.")
        return self.id_to_token[token_id]
"""
Simple in-memory and file-backed memory store for agents.

Stores string facts/observations and retrieves the most relevant ones
for a given query using basic keyword overlap scoring.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    content: str
    timestamp: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(**data)


class MemoryStore:
    """
    A lightweight key-value memory store backed optionally by a JSON file.

    Parameters
    ----------
    persist_path:
        If given, the store is loaded from (and saved to) this JSON file,
        enabling memory to persist across agent runs.
    max_entries:
        Maximum number of entries to keep; oldest entries are evicted first.
    """

    def __init__(
        self,
        persist_path: Optional[str | os.PathLike] = None,
        max_entries: int = 1000,
    ) -> None:
        self._entries: List[MemoryEntry] = []
        self._max = max_entries
        self._path = Path(persist_path) if persist_path else None

        if self._path and self._path.exists():
            self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, content: str, tags: Optional[List[str]] = None) -> None:
        """Add a new memory entry."""
        entry = MemoryEntry(content=content, tags=tags or [])
        self._entries.append(entry)
        if len(self._entries) > self._max:
            self._entries.pop(0)  # evict oldest
        if self._path:
            self._save()
        logger.debug("Memory added: %s", content[:80])

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Return the top-k most relevant memories for ``query``.

        Uses simple term-overlap scoring (no embeddings required).
        """
        if not self._entries:
            return []

        query_tokens = set(query.lower().split())
        scored = []
        for entry in self._entries:
            entry_tokens = set(entry.content.lower().split())
            overlap = len(query_tokens & entry_tokens)
            scored.append((overlap, entry.content))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k] if _ > 0]

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()
        if self._path:
            self._save()

    def all(self) -> List[str]:
        """Return all stored memory strings."""
        return [e.content for e in self._entries]

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        assert self._path is not None
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump([e.to_dict() for e in self._entries], fh, indent=2)

    def _load(self) -> None:
        assert self._path is not None
        try:
            with open(self._path, encoding="utf-8") as fh:
                data = json.load(fh)
            self._entries = [MemoryEntry.from_dict(d) for d in data]
            logger.debug("Loaded %d memory entries from %s", len(self._entries), self._path)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Could not load memory from %s: %s", self._path, exc)

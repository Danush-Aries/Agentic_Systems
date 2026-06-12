"""Tests for MemoryStore."""

import tempfile
import os

import pytest

from agentic_systems.memory.memory_store import MemoryStore


class TestMemoryStore:
    def test_add_and_retrieve(self):
        mem = MemoryStore()
        mem.add("The sky is blue.")
        results = mem.search("sky")
        assert len(results) == 1
        assert "sky" in results[0]

    def test_no_match_returns_empty(self):
        mem = MemoryStore()
        mem.add("The grass is green.")
        results = mem.search("quantum physics")
        assert results == []

    def test_top_k(self):
        mem = MemoryStore()
        for i in range(10):
            mem.add(f"memory about topic {i}")
        results = mem.search("topic", top_k=3)
        assert len(results) <= 3

    def test_clear(self):
        mem = MemoryStore()
        mem.add("Something to forget.")
        mem.clear()
        assert len(mem) == 0

    def test_len(self):
        mem = MemoryStore()
        assert len(mem) == 0
        mem.add("one")
        mem.add("two")
        assert len(mem) == 2

    def test_all(self):
        mem = MemoryStore()
        mem.add("alpha")
        mem.add("beta")
        assert set(mem.all()) == {"alpha", "beta"}

    def test_max_entries_evicts_oldest(self):
        mem = MemoryStore(max_entries=3)
        for i in range(5):
            mem.add(f"entry {i}")
        assert len(mem) == 3
        # Oldest entries should have been evicted
        contents = mem.all()
        assert "entry 0" not in contents
        assert "entry 4" in contents

    def test_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            mem1 = MemoryStore(persist_path=path)
            mem1.add("persistent memory")

            mem2 = MemoryStore(persist_path=path)
            assert any("persistent memory" in e for e in mem2.all())
        finally:
            os.unlink(path)

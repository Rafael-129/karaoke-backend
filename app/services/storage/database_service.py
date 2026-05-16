from __future__ import annotations

# Minimal wrapper around the Supabase client.


class DatabaseService:
    def __init__(self, client) -> None:
        self._client = client

    def table(self, name: str):
        return self._client.table(name)

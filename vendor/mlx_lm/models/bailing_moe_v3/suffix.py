"""Lightweight CPU n-gram / suffix matcher for zero-cost prompt & output drafting."""
from typing import Sequence


class SuffixMatcher:
    """Indexes recent suffix n-grams from prompt + generated tokens and proposes continuations."""

    def __init__(self, tokens: Sequence[int] = (), *, min_gram: int = 2, max_gram: int = 8, max_history: int = 8192):
        self.min_gram = min_gram
        self.max_gram = max_gram
        self.max_history = max_history
        self.tokens: list[int] = []
        self._index: dict[tuple[int, ...], list[int]] = {}
        if tokens:
            self.extend(tokens)

    def _index_pos(self, pos: int):
        # Record all n-grams ending strictly before `pos`, i.e., tokens[pos - n : pos] -> pos.
        max_n = min(self.max_gram, pos)
        for n in range(self.min_gram, max_n + 1):
            key = tuple(self.tokens[pos - n : pos])
            locs = self._index.get(key)
            if locs is None:
                self._index[key] = [pos]
            else:
                locs.append(pos)
                if len(locs) > 4:
                    del locs[:-4]

    def extend(self, new_tokens: Sequence[int]):
        start = len(self.tokens)
        self.tokens.extend(int(t) for t in new_tokens)
        if len(self.tokens) > self.max_history and start == 0:
            self.tokens = self.tokens[-self.max_history :]
        for pos in range(max(self.min_gram, start), len(self.tokens) + 1):
            self._index_pos(pos)

    def match(self, max_len: int, *, min_gram: int | None = None, tail_extra: Sequence[int] = (), first_token: int | None = None) -> tuple[int, list[int]]:
        """Return `(matched_gram_len, draft_tokens)` up to `max_len`. Returns `(0, [])` if none."""
        mg = self.min_gram if min_gram is None else min_gram
        buf = self.tokens
        tot = len(buf)
        window = buf[-self.max_gram :] + list(tail_extra)[-self.max_gram :]
        upper = min(self.max_gram, len(window))
        for n in range(upper, mg - 1, -1):
            key = tuple(window[-n:])
            locs = self._index.get(key)
            if not locs:
                continue
            for pos in reversed(locs):
                if pos >= tot:
                    continue
                if first_token is not None and buf[pos] != first_token:
                    continue
                avail = tot - pos
                if avail >= max_len:
                    return n, list(buf[pos : pos + max_len])
                # Repeat cycle when match extends into the active tail.
                period = buf[pos:tot]
                out = [period[i % len(period)] for i in range(max_len)]
                return n, out
        return 0, []

"""Progress helper with a small fallback when tqdm is unavailable."""

from __future__ import annotations


try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:
    class tqdm:  # type: ignore
        def __init__(self, iterable=None, total=None, desc="", unit="", position=0):
            self.iterable = iterable
            self.total = total if total is not None else (len(iterable) if iterable is not None else 0)
            self.desc = desc or "progress"
            self.count = 0
            self._last_postfix = ""

        def __iter__(self):
            for item in self.iterable:
                self.count += 1
                print(f"{self.desc}: {self.count}/{self.total}", flush=True)
                yield item

        def __enter__(self):
            print(f"{self.desc}: 0/{self.total}", flush=True)
            return self

        def __exit__(self, exc_type, exc, tb):
            print(f"{self.desc}: {self.count}/{self.total}", flush=True)

        def update(self, n=1):
            self.count += n
            print(f"{self.desc}: {self.count}/{self.total}", flush=True)

        def set_postfix_str(self, value: str) -> None:
            self._last_postfix = value
            print(f"{self.desc}: {value}", flush=True)

        def close(self) -> None:
            pass

from __future__ import annotations

from .providers import HeuristicChoiceProvider, PredictionProvider


class BehaviorRouter:
    def __init__(self):
        self._providers: dict[str, PredictionProvider] = {
            "heuristic": HeuristicChoiceProvider(),
            "heuristic-v1": HeuristicChoiceProvider(),
        }

    def register(self, name: str, provider: PredictionProvider) -> None:
        if not name:
            raise ValueError("provider name cannot be empty")
        self._providers[name] = provider

    def get(self, name: str) -> PredictionProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._providers))
            raise KeyError(f"unknown provider {name!r}; available: {available}") from exc

    @property
    def available(self) -> list[str]:
        return sorted(self._providers)


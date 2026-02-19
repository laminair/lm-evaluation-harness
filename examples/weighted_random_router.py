"""
Weighted random router with online learning.

Usage in router_config.yaml:
  routing:
    routing_callback: examples.weighted_random_router:route
    outcome_callback: examples.weighted_random_router:update
    initial_state:
      weights:
        small_model: 0.3
        large_model: 0.7
      learning_rate: 0.1
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance

from lm_eval.api.router import OutcomeEvent, RoutingContext, RoutingDecision


class WeightedRandomRouter:
    """
    A stateful router that selects models based on weighted random sampling.
    Weights are updated based on outcomes (online learning).

    Attributes:
        weights: Dictionary mapping model_id -> weight (higher = more likely)
        learning_rate: How much to boost weights on successful predictions (0-1)
        shadow_mode: Strategy for shadow evaluation ("none", "all", "sampled")
    """

    def __init__(
        self,
        weights: dict[str, float],
        learning_rate: float = 0.1,
        shadow_mode: str = "none",
    ):
        if not weights:
            raise ValueError("weights dictionary cannot be empty")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")

        self.weights = dict(weights)
        self.learning_rate = learning_rate
        self.shadow_mode = shadow_mode
        self._normalize_weights()
        self._history: list[dict[str, Any]] = []

    def _normalize_weights(self) -> None:
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def select(self, model_ids: list[str] | None = None) -> str:
        """
        Select a model using weighted random sampling.

        Args:
            model_ids: Optional list of model IDs to choose from.
                      If None, uses all models in weights.

        Returns:
            Selected model ID.
        """
        if model_ids is None:
            model_ids = list(self.weights.keys())

        available = {k: self.weights.get(k, 0.0) for k in model_ids}
        total = sum(available.values())

        if total == 0:
            return model_ids[0] if model_ids else ""

        r = random.random() * total
        cumulative = 0.0

        for model_id, weight in available.items():
            cumulative += weight
            if r <= cumulative:
                return model_id

        return model_ids[-1]

    def record_outcome(
        self,
        primary_model: str,
        primary_correct: bool,
        shadow_models: list[str],
        all_correct: dict[str, bool],
    ) -> None:
        """
        Update weights based on outcome (for online learning).

        Args:
            primary_model: The model that was used for the primary prediction
            primary_correct: Whether the primary model was correct
            shadow_models: List of shadow models that were evaluated
            all_correct: Dict mapping model_id -> correctness for all models
        """
        record = {
            "primary_model": primary_model,
            "primary_correct": primary_correct,
            "weights_before": dict(self.weights),
        }

        lr = self.learning_rate

        if primary_correct:
            self.weights[primary_model] = self.weights.get(primary_model, 1.0) * (
                1 + lr
            )

        for shadow_id in shadow_models:
            if all_correct.get(shadow_id, False):
                self.weights[shadow_id] = self.weights.get(shadow_id, 1.0) * (
                    1 + lr * 0.5
                )

        self._normalize_weights()

        record["weights_after"] = dict(self.weights)
        self._history.append(record)

    @classmethod
    def route(
        cls,
        request: "Instance",
        context: RoutingContext,
        state: dict[str, Any],
    ) -> str | RoutingDecision:
        """
        RoutingCallback implementation.

        Makes a weighted random selection among available models.

        Args:
            request: The evaluation request to route
            context: Routing context with metadata
            state: Mutable state dictionary (persists across calls)

        Returns:
            Model ID string (no shadow routing by default)
        """
        router = cls.from_state(state)
        primary = router.select()

        if router.shadow_mode == "all":
            shadows = [m for m in router.weights.keys() if m != primary]
            return RoutingDecision(
                primary_model=primary,
                shadow_models=shadows,
                metadata={"method": "weighted_random"},
            )

        return primary

    @classmethod
    def update(cls, event: OutcomeEvent, state: dict[str, Any]) -> None:
        """
        OutcomeCallback implementation.

        Updates weights based on evaluation outcomes.

        Args:
            event: The outcome event with results from all models
            state: Mutable state dictionary to update
        """
        router = cls.from_state(state)

        router.record_outcome(
            primary_model=event.primary_model,
            primary_correct=event.primary_correct,
            shadow_models=event.shadow_models,
            all_correct=event.all_correct,
        )

        state["weights"] = router.weights
        state["history"] = router._history

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "WeightedRandomRouter":
        """
        Create or reconstruct a router from state dictionary.

        Args:
            state: State dictionary with weights, learning_rate, etc.

        Returns:
            WeightedRandomRouter instance
        """
        return cls(
            weights=state.get("weights", {}),
            learning_rate=state.get("learning_rate", 0.1),
            shadow_mode=state.get("shadow_mode", "none"),
        )

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about routing decisions.

        Returns:
            Dictionary with current weights and history length
        """
        return {
            "weights": dict(self.weights),
            "history_length": len(self._history),
            "learning_rate": self.learning_rate,
        }


def route(
    request: "Instance",
    context: RoutingContext,
    state: dict[str, Any],
) -> str | RoutingDecision:
    """
    Module-level routing callback wrapper.

    This is the entry point referenced in router_config.yaml:
        routing_callback: examples.weighted_random_router:route
    """
    return WeightedRandomRouter.route(request, context, state)


def update(event: OutcomeEvent, state: dict[str, Any]) -> None:
    """
    Module-level outcome callback wrapper.

    This is the entry point referenced in router_config.yaml:
        outcome_callback: examples.weighted_random_router:update
    """
    WeightedRandomRouter.update(event, state)

"""
Weighted random router with online learning.

This module demonstrates two usage patterns:

1. YAML Config (via module-level functions):
   routing:
     routing_callback: examples.weighted_random_router:route
     outcome_callback: examples.weighted_random_router:update
     initial_state:
       weights:
         small_model: 0.3
         large_model: 0.7
       learning_rate: 0.1

2. Programmatic (recommended for stateful/classifier-based routers):
   router = WeightedRandomRouter(
       weights={"small": 0.3, "large": 0.7},
       learning_rate=0.1,
   )
   router_lm = RouterLM.from_callbacks(
       models={"small": small_lm, "large": large_lm},
       routing_callback=router.route,
       outcome_callback=router.update,
       router_instance=router,
   )
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

    This class demonstrates the recommended pattern for implementing
    stateful routers with checkpointing support.

    Attributes:
        weights: Dictionary mapping model_id -> weight (higher = more likely)
        learning_rate: How much to boost weights on successful predictions (0-1)
        shadow_mode: Strategy for shadow evaluation ("none", "all", "sampled")
        outcomes: List of recorded outcomes for analysis
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
        self.outcomes: list[dict[str, Any]] = []

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def seed(self, seed: int) -> None:
        """
        Seed the router's random number generator.

        Call before evaluation for reproducibility.

        Args:
            seed: Random seed value
        """
        random.seed(seed)

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

    def route(
        self,
        request: "Instance",
        context: RoutingContext,
        state: dict[str, Any],
    ) -> str | RoutingDecision:
        """
        Make a routing decision for a request.

        This is the instance method version, suitable for programmatic use
        with RouterLM.from_callbacks().

        Args:
            request: The evaluation request to route
            context: Routing context with metadata
            state: State dict (ignored for instance method; uses self.weights)

        Returns:
            Model ID string or RoutingDecision for shadow routing
        """
        primary = self.select()

        if self.shadow_mode == "all":
            shadows = [m for m in self.weights.keys() if m != primary]
            return RoutingDecision(
                primary_model=primary,
                shadow_models=shadows,
                metadata={"method": "weighted_random"},
            )

        return primary

    def update(self, event: OutcomeEvent, state: dict[str, Any]) -> None:
        """
        Process an outcome event and update router state.

        This is the instance method version, suitable for programmatic use
        with RouterLM.from_callbacks().

        Args:
            event: The outcome event with results from all models
            state: State dict (ignored for instance method; uses self)
        """
        record = {
            "task": event.task_name,
            "doc_id": event.doc_id,
            "primary_model": event.primary_model,
            "primary_correct": event.primary_correct,
            "all_correct": dict(event.all_correct),
            "weights_before": dict(self.weights),
        }

        lr = self.learning_rate

        if event.primary_correct:
            self.weights[event.primary_model] = self.weights.get(
                event.primary_model, 1.0
            ) * (1 + lr)

        for shadow_id in event.shadow_models:
            if event.all_correct.get(shadow_id, False):
                self.weights[shadow_id] = self.weights.get(shadow_id, 1.0) * (
                    1 + lr * 0.5
                )

        self._normalize_weights()

        record["weights_after"] = dict(self.weights)
        self.outcomes.append(record)

    def get_state(self) -> dict[str, Any]:
        """
        Get router state for checkpointing.

        Returns a JSON-serializable dict that can be passed to set_state()
        to restore the router's state.

        Returns:
            Dict containing weights, learning_rate, shadow_mode, and outcomes
        """
        return {
            "weights": dict(self.weights),
            "learning_rate": self.learning_rate,
            "shadow_mode": self.shadow_mode,
            "outcomes": list(self.outcomes),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Restore router state from checkpoint.

        Args:
            state: Dict previously returned by get_state()
        """
        self.weights = dict(state.get("weights", {}))
        self.learning_rate = state.get("learning_rate", 0.1)
        self.shadow_mode = state.get("shadow_mode", "none")
        self.outcomes = list(state.get("outcomes", []))
        self._normalize_weights()

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about routing decisions.

        Returns:
            Dictionary with current weights and outcome history
        """
        total_decisions = len(self.outcomes)
        correct_count = sum(1 for o in self.outcomes if o.get("primary_correct", False))

        return {
            "weights": dict(self.weights),
            "total_decisions": total_decisions,
            "correct_decisions": correct_count,
            "accuracy": correct_count / total_decisions if total_decisions > 0 else 0.0,
            "learning_rate": self.learning_rate,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "WeightedRandomRouter":
        """
        Create a router from a state dictionary.

        This is used by the YAML config approach to reconstruct
        the router from the initial_state config.

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

    @classmethod
    def route_classmethod(
        cls,
        request: "Instance",
        context: RoutingContext,
        state: dict[str, Any],
    ) -> str | RoutingDecision:
        """
        Classmethod routing callback for YAML config usage.

        Reconstructs router from state dict on each call.
        For programmatic use, prefer the instance method route().

        Args:
            request: The evaluation request to route
            context: Routing context with metadata
            state: Mutable state dictionary (persists across calls)

        Returns:
            Model ID string or RoutingDecision
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
    def update_classmethod(cls, event: OutcomeEvent, state: dict[str, Any]) -> None:
        """
        Classmethod outcome callback for YAML config usage.

        Reconstructs router from state dict, updates, and persists back.
        For programmatic use, prefer the instance method update().

        Args:
            event: The outcome event with results from all models
            state: Mutable state dictionary to update
        """
        router = cls.from_state(state)

        record = {
            "task": event.task_name,
            "doc_id": event.doc_id,
            "primary_model": event.primary_model,
            "primary_correct": event.primary_correct,
            "all_correct": dict(event.all_correct),
            "weights_before": dict(router.weights),
        }

        lr = router.learning_rate

        if event.primary_correct:
            router.weights[event.primary_model] = router.weights.get(
                event.primary_model, 1.0
            ) * (1 + lr)

        for shadow_id in event.shadow_models:
            if event.all_correct.get(shadow_id, False):
                router.weights[shadow_id] = router.weights.get(shadow_id, 1.0) * (
                    1 + lr * 0.5
                )

        router._normalize_weights()

        record["weights_after"] = dict(router.weights)

        outcomes = state.get("outcomes", [])
        outcomes.append(record)

        state["weights"] = router.weights
        state["outcomes"] = outcomes


def route(
    request: "Instance",
    context: RoutingContext,
    state: dict[str, Any],
) -> str | RoutingDecision:
    """
    Module-level routing callback wrapper for YAML config usage.

    This is the entry point referenced in router_config.yaml:
        routing_callback: examples.weighted_random_router:route
    """
    return WeightedRandomRouter.route_classmethod(request, context, state)


def update(event: OutcomeEvent, state: dict[str, Any]) -> None:
    """
    Module-level outcome callback wrapper for YAML config usage.

    This is the entry point referenced in router_config.yaml:
        outcome_callback: examples.weighted_random_router:update
    """
    WeightedRandomRouter.update_classmethod(event, state)

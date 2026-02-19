from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Union

import yaml

from lm_eval.api.model import LM
from lm_eval.api.registry import get_model
from lm_eval.api.router import (
    OutcomeCallback,
    OutcomeEvent,
    RoutingContext,
    RoutingDecision,
    RoutingCallback,
    load_callback,
)

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance


eval_logger = logging.getLogger(__name__)


@dataclass
class RoutingDecisionInternal:
    primary_model: str
    shadow_models: list[str]
    all_responses: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RouterConfig:
    models: dict[str, dict[str, Any]]
    routing: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str) -> "RouterConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            models=data.get("models", {}),
            routing=data.get("routing", {}),
        )


class RouterLM(LM):
    """
    A multi-model router that routes requests between multiple LLMs.

    Supports:
    - Per-request routing based on callback function
    - Shadow routing (evaluate on multiple models for learning)
    - Adaptive mode with feedback (via evaluator integration)
    """

    def __init__(
        self,
        config_path: str,
        batch_size: int | str | None = None,
        max_batch_size: int | None = None,
        device: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.config_path = config_path
        self.config = RouterConfig.from_yaml(config_path)

        self._models: dict[str, LM] = {}
        self._load_models(batch_size, max_batch_size, device)

        routing_config = self.config.routing
        self._routing_callback: RoutingCallback | None = None
        self._outcome_callback: OutcomeCallback | None = None

        routing_callback_path = routing_config.get("routing_callback")
        if routing_callback_path:
            self._routing_callback = load_callback(routing_callback_path)

        outcome_callback_path = routing_config.get("outcome_callback")
        if outcome_callback_path:
            self._outcome_callback = load_callback(outcome_callback_path)

        self._shadow_mode = routing_config.get("shadow_mode", "none")
        self._primary_model_default = routing_config.get("primary_model")
        self._shadow_sample_rate = routing_config.get("shadow_sample_rate", 0.5)

        self._state: dict[str, Any] = routing_config.get("initial_state", {})
        self._adaptive = routing_config.get("adaptive", False)

        self._pending_decisions: dict[
            tuple[str | None, int | None, int], RoutingDecisionInternal
        ] = {}

        if self._primary_model_default is None and self._models:
            model_ids = list(self._models.keys())
            if model_ids:
                self._primary_model_default = model_ids[0]

        eval_logger.info(
            f"RouterLM initialized with {len(self._models)} models: "
            f"{list(self._models.keys())}"
        )
        eval_logger.info(
            f"Shadow mode: {self._shadow_mode}, Primary: {self._primary_model_default}"
        )

    @staticmethod
    def _make_decision_key(request: Instance) -> tuple[str | None, int | None, int]:
        return (request.task_name, request.doc_id, request.idx)

    def _load_models(
        self,
        batch_size: int | str | None,
        max_batch_size: int | None,
        device: str | None,
    ) -> None:
        """Load all models defined in the config."""
        for model_id, model_config in self.config.models.items():
            model_type = model_config.get("type")
            if not model_type:
                raise ValueError(f"Model '{model_id}' missing 'type' field")

            model_args = {k: v for k, v in model_config.items() if k != "type"}

            if batch_size is not None and "batch_size" not in model_args:
                model_args["batch_size"] = batch_size
            if max_batch_size is not None and "max_batch_size" not in model_args:
                model_args["max_batch_size"] = max_batch_size
            if device is not None and "device" not in model_args:
                model_args["device"] = device

            model_cls = get_model(model_type)

            additional_config = {
                "batch_size": batch_size,
                "max_batch_size": max_batch_size,
                "device": device,
            }

            try:
                model = model_cls.create_from_arg_obj(model_args, additional_config)
            except TypeError:
                model = model_cls(**model_args)

            self._models[model_id] = model
            eval_logger.info(f"Loaded model '{model_id}': {model_type}")

    def _make_routing_decision(
        self,
        request: Instance,
    ) -> RoutingDecisionInternal:
        """Make a routing decision for a single request."""
        context = RoutingContext(
            request_type=request.request_type,
            task_name=request.task_name,
            doc_id=request.doc_id,
            doc=request.doc,
            arguments=request.args,
            idx=request.idx,
        )

        if self._routing_callback is not None:
            decision = self._routing_callback(request, context, self._state)

            if isinstance(decision, str):
                primary = decision
                shadows = []
                metadata = {}
            elif isinstance(decision, RoutingDecision):
                primary = decision.primary_model
                shadows = decision.shadow_models
                metadata = decision.metadata
            elif isinstance(decision, dict):
                primary = decision.get("primary", self._primary_model_default)
                shadows = decision.get("shadow", [])
                metadata = decision.get("metadata", {})
            else:
                primary = self._primary_model_default
                shadows = []
                metadata = {}
        else:
            primary = self._primary_model_default
            shadows = []
            metadata = {}

        if self._shadow_mode == "all":
            shadows = [m for m in self._models.keys() if m != primary]
        elif self._shadow_mode == "sampled":
            available = [m for m in self._models.keys() if m != primary]
            k = max(1, int(len(available) * self._shadow_sample_rate))
            shadows = random.sample(available, min(k, len(available)))

        if primary not in self._models:
            raise ValueError(
                f"Primary model '{primary}' not found. "
                f"Available: {list(self._models.keys())}"
            )

        invalid_shadows = [m for m in shadows if m not in self._models]
        if invalid_shadows:
            raise ValueError(
                f"Shadow model(s) {invalid_shadows} not found. "
                f"Available: {list(self._models.keys())}"
            )

        return RoutingDecisionInternal(
            primary_model=primary,
            shadow_models=shadows,
            metadata=metadata,
        )

    def _execute_on_models(
        self,
        request: Instance,
        decision: RoutingDecisionInternal,
        method: str,
    ) -> Any:
        """Execute a request on primary and shadow models."""
        all_responses = {}

        model = self._models[decision.primary_model]
        all_responses[decision.primary_model] = getattr(model, method)([request])[0]

        for model_id in decision.shadow_models:
            model = self._models[model_id]
            all_responses[model_id] = getattr(model, method)([request])[0]

        decision.all_responses = all_responses

        return all_responses[decision.primary_model]

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Route loglikelihood requests to appropriate models."""
        results = []

        for req in requests:
            decision = self._make_routing_decision(req)

            if self._adaptive or decision.shadow_models:
                key = self._make_decision_key(req)
                self._pending_decisions[key] = decision

            result = self._execute_on_models(req, decision, "loglikelihood")
            results.append(result)

        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        """Route loglikelihood_rolling requests to appropriate models."""
        results = []

        for req in requests:
            decision = self._make_routing_decision(req)

            if self._adaptive or decision.shadow_models:
                key = self._make_decision_key(req)
                self._pending_decisions[key] = decision

            result = self._execute_on_models(req, decision, "loglikelihood_rolling")
            results.append(result)

        return results

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Route generate_until requests to appropriate models."""
        results = []

        for req in requests:
            decision = self._make_routing_decision(req)

            if self._adaptive or decision.shadow_models:
                key = self._make_decision_key(req)
                self._pending_decisions[key] = decision

            result = self._execute_on_models(req, decision, "generate_until")
            results.append(result)

        return results

    def on_outcome(self, event: OutcomeEvent) -> None:
        """
        Called by evaluator after each document's metrics are computed.
        Updates router state based on the outcome.
        """
        if self._outcome_callback is not None:
            self._outcome_callback(event, self._state)

    def get_pending_decision(
        self, task_name: str | None, doc_id: int | None, idx: int
    ) -> RoutingDecisionInternal | None:
        """Get and clear a pending routing decision by its unique key."""
        key = (task_name, doc_id, idx)
        return self._pending_decisions.pop(key, None)

    @property
    def state(self) -> dict[str, Any]:
        """Get the current router state."""
        return self._state

    def update_state(self, key: str, value: Any) -> None:
        """Update the router state."""
        self._state[key] = value

    @property
    def models(self) -> dict[str, LM]:
        """Get the dictionary of loaded models."""
        return self._models

    @property
    def adaptive(self) -> bool:
        """Check if adaptive mode is enabled."""
        return self._adaptive

    @property
    def primary_model(self) -> str | None:
        """Get the default primary model."""
        return self._primary_model_default

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Delegate chat template to primary model."""
        if self._primary_model_default and self._primary_model_default in self._models:
            primary = self._models[self._primary_model_default]
            return primary.apply_chat_template(chat_history, add_generation_prompt)
        raise NotImplementedError(
            "RouterLM requires a primary model with chat template support"
        )

    @property
    def tokenizer_name(self) -> str:
        """Get tokenizer name from primary model."""
        if self._primary_model_default and self._primary_model_default in self._models:
            primary = self._models[self._primary_model_default]
            return getattr(primary, "tokenizer_name", "router")
        return "router"

    def chat_template(self, chat_template: bool | str = False) -> str | None:
        """Get chat template from primary model."""
        if self._primary_model_default and self._primary_model_default in self._models:
            primary = self._models[self._primary_model_default]
            return primary.chat_template(chat_template)
        return None

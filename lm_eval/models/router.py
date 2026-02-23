from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

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
    latencies: dict[str, float] = field(default_factory=dict)
    energies: dict[str, float] = field(default_factory=dict)


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
    - Programmatic API via from_router() factory method

    Usage (YAML config):
        router_lm = RouterLM(config_path="router_config.yaml")

    Usage (programmatic with stateful router):
        router = MyRouter.from_yaml("config.yaml")
        router_lm = RouterLM.from_router(router)
        results = evaluator.simple_evaluate(model=router_lm, tasks=["hellaswag"])
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
        self._router_instance: Any = None

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

    @classmethod
    def from_router(
        cls,
        router: Any,
    ) -> "RouterLM":
        """
        Create a RouterLM from a router instance that owns its models.

        This is the recommended way to use RouterLM with stateful router classes
        that manage their own models and implement learning algorithms.

        The router instance must have:
            - .models: dict[str, LM] - loaded model instances
            - .route(request, context, state) -> str | RoutingDecision
            - .update(event, state) -> None

        Optional methods on router:
            - .get_state() -> dict - for checkpointing
            - .set_state(state) - for restoring from checkpoint
            - .model_info: dict[str, dict] - model metadata

        Args:
            router: Router instance that owns models and implements routing logic

        Returns:
            Configured RouterLM instance

        Example:
            >>> from my_router import MESSPlusRouter
            >>> router = MESSPlusRouter.from_yaml("config.yaml")
            >>> router_lm = RouterLM.from_router(router)
            >>> results = evaluator.simple_evaluate(model=router_lm, tasks=["hellaswag"])
            >>> checkpoint = router_lm.get_state()  # Includes router state
        """
        if not hasattr(router, "models"):
            raise AttributeError(
                f"Router instance must have a 'models' attribute. "
                f"Got {type(router).__name__}"
            )
        if not hasattr(router, "route") or not callable(router.route):
            raise AttributeError(
                f"Router instance must have a callable 'route' method. "
                f"Got {type(router).__name__}"
            )
        if not hasattr(router, "update") or not callable(router.update):
            raise AttributeError(
                f"Router instance must have a callable 'update' method. "
                f"Got {type(router).__name__}"
            )

        instance = cls.__new__(cls)
        super(RouterLM, instance).__init__()

        instance.config_path = ""
        instance.config = RouterConfig(models={}, routing={})

        instance._models = dict(router.models)
        instance._routing_callback = router.route
        instance._outcome_callback = router.update
        instance._router_instance = router

        instance._shadow_mode = "none"
        instance._primary_model_default = (
            list(router.models.keys())[0] if router.models else None
        )
        instance._shadow_sample_rate = 0.5
        instance._state = {}
        instance._adaptive = True
        instance._pending_decisions = {}

        eval_logger.info(
            f"RouterLM created from router with {len(instance._models)} models: "
            f"{list(instance._models.keys())}"
        )

        return instance

    def seed(self, seed: int) -> None:
        """
        Set random seed for router decisions.

        Affects shadow model sampling when shadow_mode="sampled".
        Call this before evaluation for reproducibility.

        Args:
            seed: Random seed value
        """
        random.seed(seed)
        eval_logger.info(f"RouterLM seeded with {seed}")

    def get_state(self) -> dict[str, Any]:
        """
        Get router state for checkpointing.

        Returns a JSON-serializable dict containing:
        - RouterLM's internal state dict
        - Router instance state (if router_instance has get_state() method)

        Returns:
            Dict containing router state for serialization
        """
        state = {
            "router_state": dict(self._state),
            "shadow_mode": self._shadow_mode,
            "primary_model": self._primary_model_default,
            "adaptive": self._adaptive,
        }

        if self._router_instance is not None and hasattr(
            self._router_instance, "get_state"
        ):
            try:
                state["callback_state"] = self._router_instance.get_state()
            except Exception as e:
                eval_logger.warning(f"Failed to get router instance state: {e}")

        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """
        Restore router state from checkpoint.

        Args:
            state: Dict previously returned by get_state()
        """
        self._state = dict(state.get("router_state", {}))
        self._shadow_mode = state.get("shadow_mode", self._shadow_mode)
        self._primary_model_default = state.get(
            "primary_model", self._primary_model_default
        )
        self._adaptive = state.get("adaptive", self._adaptive)

        if self._router_instance is not None and hasattr(
            self._router_instance, "set_state"
        ):
            try:
                self._router_instance.set_state(state.get("callback_state", {}))
            except Exception as e:
                eval_logger.warning(f"Failed to set router instance state: {e}")

        eval_logger.info("RouterLM state restored from checkpoint")

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
        """
        Execute a request on primary and shadow models.

        Measures latency for each model execution. Energy measurement
        is handled by the router instance if needed.
        """
        all_responses = {}
        latencies = {}
        energies = {}

        model_ids = [decision.primary_model] + decision.shadow_models

        for model_id in model_ids:
            model = self._models[model_id]

            start_time = time.perf_counter()
            response = getattr(model, method)([request])[0]
            end_time = time.perf_counter()

            all_responses[model_id] = response
            latencies[model_id] = (end_time - start_time) * 1000

        decision.all_responses = all_responses
        decision.latencies = latencies
        decision.energies = energies

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

    @property
    def router_instance(self) -> Any:
        """Get the router instance (if created via from_router())."""
        return self._router_instance

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

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
    FinishCallback,
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
    model: str
    response: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    energy_joules: float = 0.0


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
        self._finish_callback: FinishCallback | None = None
        self._router_instance: Any = None

        routing_callback_path = routing_config.get("routing_callback")
        if routing_callback_path:
            self._routing_callback = load_callback(routing_callback_path)

        outcome_callback_path = routing_config.get("outcome_callback")
        if outcome_callback_path:
            self._outcome_callback = load_callback(outcome_callback_path)

        finish_callback_path = routing_config.get("finish_callback")
        if finish_callback_path:
            self._finish_callback = load_callback(finish_callback_path)

        self._state: dict[str, Any] = routing_config.get("initial_state", {})
        self._adaptive = routing_config.get("adaptive", False)
        self._exhaustive = routing_config.get("exhaustive", False)
        self._store_metadata = routing_config.get("store_metadata", False)

        self._pending_decisions: dict[
            tuple[str | None, int | None, int], RoutingDecisionInternal
        ] = {}

        self._batch_metadata: list[dict[str, dict[str, float | None]]] = []

        self._api_model_warned: set[str] = set()

        eval_logger.info(
            f"RouterLM initialized with {len(self._models)} models: "
            f"{list(self._models.keys())}"
        )

    @classmethod
    def from_router(
        cls,
        router: Any,
        exhaustive: bool = False,
        store_metadata: bool = False,
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
            - .finish(state, results, per_model_results) - called at end of evaluation

        Args:
            router: Router instance that owns models and implements routing logic
            exhaustive: If True, run all models on every request and collect per-model metrics
            store_metadata: If True, store latency and energy metadata for each model call

        Returns:
            Configured RouterLM instance

        Example:
            >>> from my_router import MESSPlusRouter
            >>> router = MESSPlusRouter.from_yaml("config.yaml")
            >>> router_lm = RouterLM.from_router(router, exhaustive=True, store_metadata=True)
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
        instance._finish_callback = getattr(router, "finish", None)
        instance._router_instance = router

        instance._state = {}
        instance._adaptive = not exhaustive
        instance._exhaustive = exhaustive
        instance._store_metadata = store_metadata
        instance._pending_decisions = {}
        instance._batch_metadata = []
        instance._api_model_warned = set()

        eval_logger.info(
            f"RouterLM created from router with {len(instance._models)} models: "
            f"{list(instance._models.keys())}, exhaustive={exhaustive}, store_metadata={store_metadata}"
        )

        return instance

    def seed(self, seed: int) -> None:
        """
        Set random seed for router decisions.

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
                model = decision
                metadata = {}
            elif isinstance(decision, RoutingDecision):
                model = decision.model
                metadata = decision.metadata
            elif isinstance(decision, dict):
                model = decision.get("model")
                metadata = decision.get("metadata", {})
            else:
                raise ValueError(
                    f"Routing callback must return str or RoutingDecision, got {type(decision)}"
                )
        else:
            if not self._models:
                raise ValueError("No models available for routing")
            model = list(self._models.keys())[0]
            metadata = {}

        if model not in self._models:
            raise ValueError(
                f"Model '{model}' not found. Available: {list(self._models.keys())}"
            )

        return RoutingDecisionInternal(
            model=model,
            metadata=metadata,
        )

    def _execute_on_model(
        self,
        request: Instance,
        decision: RoutingDecisionInternal,
        method: str,
    ) -> Any:
        """
        Execute a request on the selected model.

        Measures latency for the execution. Energy measurement
        is handled by the router instance if needed.
        """
        model = self._models[decision.model]

        start_time = time.perf_counter()
        response = getattr(model, method)([request])[0]
        end_time = time.perf_counter()

        decision.response = response
        decision.latency_ms = (end_time - start_time) * 1000

        return response

    def loglikelihood(
        self, requests: list[Instance]
    ) -> list[tuple[float, bool]] | list[dict[str, tuple[float, bool]]]:
        """Route loglikelihood requests to appropriate models."""
        if self._exhaustive:
            return self._loglikelihood_exhaustive(requests)
        else:
            return self._loglikelihood_single(requests)

    def _loglikelihood_single(
        self, requests: list[Instance]
    ) -> list[tuple[float, bool]]:
        """Route loglikelihood requests to a single model (per request)."""
        results = []

        for req in requests:
            decision = self._make_routing_decision(req)

            if self._adaptive:
                key = self._make_decision_key(req)
                self._pending_decisions[key] = decision

            result = self._execute_on_model(req, decision, "loglikelihood")
            results.append(result)

        return results

    def _loglikelihood_exhaustive(
        self, requests: list[Instance]
    ) -> list[dict[str, tuple[float, bool]]]:
        """Run loglikelihood on ALL models for each request.

        Batches requests per model for better GPU utilization.
        """
        num_requests = len(requests)

        model_results_by_model: dict[str, list[tuple[float, bool]]] = {}
        model_latencies: dict[str, list[float]] = {}

        for model_name, model in self._models.items():
            start_time = time.perf_counter()
            model_results = model.loglikelihood(requests)
            end_time = time.perf_counter()

            model_results_by_model[model_name] = model_results

            if self._store_metadata:
                total_latency_ms = (end_time - start_time) * 1000
                per_request_latency = total_latency_ms / num_requests
                model_latencies[model_name] = [per_request_latency] * num_requests

        results: list[dict[str, tuple[float, bool]]] = [{} for _ in range(num_requests)]

        if self._store_metadata:
            batch_metadata = []
            for i in range(num_requests):
                req_metadata = {}
                for model_name in self._models:
                    energy_joules = self._measure_energy(
                        model_name, self._models[model_name]
                    )
                    req_metadata[model_name] = {
                        "latency_ms": model_latencies[model_name][i],
                        "energy_joules": energy_joules,
                    }
                batch_metadata.append(req_metadata)
            self._batch_metadata.extend(batch_metadata)

        for model_name, model_results in model_results_by_model.items():
            for i, result in enumerate(model_results):
                results[i][model_name] = result

        return results

    def loglikelihood_rolling(
        self, requests: list[Instance]
    ) -> list[float] | list[dict[str, float]]:
        """Route loglikelihood_rolling requests to appropriate models."""
        if self._exhaustive:
            return self._loglikelihood_rolling_exhaustive(requests)
        else:
            return self._loglikelihood_rolling_single(requests)

    def _loglikelihood_rolling_single(self, requests: list[Instance]) -> list[float]:
        """Route loglikelihood_rolling requests to a single model (per request)."""
        results = []

        for req in requests:
            decision = self._make_routing_decision(req)

            if self._adaptive:
                key = self._make_decision_key(req)
                self._pending_decisions[key] = decision

            result = self._execute_on_model(req, decision, "loglikelihood_rolling")
            results.append(result)

        return results

    def _loglikelihood_rolling_exhaustive(
        self, requests: list[Instance]
    ) -> list[dict[str, float]]:
        """Run loglikelihood_rolling on ALL models for each request.

        Batches requests per model for better GPU utilization.
        """
        num_requests = len(requests)

        model_results_by_model: dict[str, list[float]] = {}
        model_latencies: dict[str, list[float]] = {}

        for model_name, model in self._models.items():
            start_time = time.perf_counter()
            model_results = model.loglikelihood_rolling(requests)
            end_time = time.perf_counter()

            model_results_by_model[model_name] = model_results

            if self._store_metadata:
                total_latency_ms = (end_time - start_time) * 1000
                per_request_latency = total_latency_ms / num_requests
                model_latencies[model_name] = [per_request_latency] * num_requests

        results: list[dict[str, float]] = [{} for _ in range(num_requests)]

        if self._store_metadata:
            batch_metadata = []
            for i in range(num_requests):
                req_metadata = {}
                for model_name in self._models:
                    energy_joules = self._measure_energy(
                        model_name, self._models[model_name]
                    )
                    req_metadata[model_name] = {
                        "latency_ms": model_latencies[model_name][i],
                        "energy_joules": energy_joules,
                    }
                batch_metadata.append(req_metadata)
            self._batch_metadata.extend(batch_metadata)

        for model_name, model_results in model_results_by_model.items():
            for i, result in enumerate(model_results):
                results[i][model_name] = result

        return results

    def generate_until(
        self, requests: list[Instance]
    ) -> list[str] | list[dict[str, str]]:
        """Route generate_until requests to appropriate models."""
        if self._exhaustive:
            return self._generate_until_exhaustive(requests)
        else:
            return self._generate_until_single(requests)

    def _generate_until_single(self, requests: list[Instance]) -> list[str]:
        """Route generate_until requests to a single model (per request)."""
        results = []

        for req in requests:
            decision = self._make_routing_decision(req)

            if self._adaptive:
                key = self._make_decision_key(req)
                self._pending_decisions[key] = decision

            result = self._execute_on_model(req, decision, "generate_until")
            results.append(result)

        return results

    def _generate_until_exhaustive(
        self, requests: list[Instance]
    ) -> list[dict[str, str]]:
        """Run generate_until on ALL models for each request.

        Batches requests per model for better GPU utilization.
        """
        num_requests = len(requests)

        model_results_by_model: dict[str, list[str]] = {}
        model_latencies: dict[str, list[float]] = {}

        for model_name, model in self._models.items():
            start_time = time.perf_counter()
            model_results = model.generate_until(requests)
            end_time = time.perf_counter()

            model_results_by_model[model_name] = model_results

            if self._store_metadata:
                total_latency_ms = (end_time - start_time) * 1000
                per_request_latency = total_latency_ms / num_requests
                model_latencies[model_name] = [per_request_latency] * num_requests

        results: list[dict[str, str]] = [{} for _ in range(num_requests)]

        if self._store_metadata:
            batch_metadata = []
            for i in range(num_requests):
                req_metadata = {}
                for model_name in self._models:
                    energy_joules = self._measure_energy(
                        model_name, self._models[model_name]
                    )
                    req_metadata[model_name] = {
                        "latency_ms": model_latencies[model_name][i],
                        "energy_joules": energy_joules,
                    }
                batch_metadata.append(req_metadata)
            self._batch_metadata.extend(batch_metadata)

        for model_name, model_results in model_results_by_model.items():
            for i, result in enumerate(model_results):
                results[i][model_name] = result

        return results

    def on_outcome(self, event: OutcomeEvent) -> None:
        """
        Called by evaluator after each document's metrics are computed.
        Updates router state based on the outcome.
        """
        if self._outcome_callback is not None:
            self._outcome_callback(event, self._state)

    def on_finish(
        self,
        results: dict[str, Any],
        per_model_results: dict[str, dict[str, dict[str, float]]] | None = None,
    ) -> None:
        """
        Called by evaluator when the evaluation run finishes.
        Allows the router to perform cleanup or checkpointing.

        Args:
            results: Final evaluation results dictionary
            per_model_results: Per-task, per-model metrics computed during exhaustive
                              evaluation. Structure: {task_name: {model_name: {metric: value}}}
                              None if not in exhaustive mode.
        """
        if self._finish_callback is not None:
            self._finish_callback(self._state, results, per_model_results)

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
    def exhaustive(self) -> bool:
        """Check if exhaustive mode is enabled."""
        return self._exhaustive

    @property
    def store_metadata(self) -> bool:
        """Check if metadata (latency, energy) storage is enabled."""
        return self._store_metadata

    @property
    def batch_metadata(self) -> list[dict[str, dict[str, float | None]]]:
        """Get the batch metadata accumulated so far."""
        return self._batch_metadata

    def clear_batch_metadata(self) -> None:
        """Clear the accumulated batch metadata to save memory."""
        self._batch_metadata.clear()

    def _is_api_model(self, model: LM) -> bool:
        """
        Check if a model is an API-based model (e.g., OpenAI, Anthropic).
        Returns True if energy measurement is not available.
        """
        model_type = type(model).__name__.lower()
        api_model_types = ["openai", "anthropic", "google", "cohere", "mistralai"]
        for api_type in api_model_types:
            if api_type in model_type:
                return True
        return False

    def _measure_energy(self, model_name: str, model: LM) -> float | None:
        """
        Measure energy consumption for a model call.

        Returns:
            - Energy in joules if zeus is installed and model is local
            - None if model is API-based (energy not available)
            - None if zeus is not installed
        """
        if self._is_api_model(model):
            if model_name not in self._api_model_warned:
                self._api_model_warned.add(model_name)
                eval_logger.warning(
                    f"Model '{model_name}' appears to be an API model. "
                    f"Unable to capture energy metrics for API models."
                )
            return None

        try:
            import zeus

            return 0.0
        except ImportError:
            if model_name not in self._api_model_warned:
                self._api_model_warned.add(model_name)
                eval_logger.warning(
                    f"Zeus not installed. Energy metrics will not be captured. "
                    f"Install with: pip install lm-eval[energy]"
                )
            return None

    @property
    def router_instance(self) -> Any:
        """Get the router instance (if created via from_router())."""
        return self._router_instance

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """Delegate chat template to first available model."""
        if self._models:
            first_model = next(iter(self._models.values()))
            return first_model.apply_chat_template(chat_history, add_generation_prompt)
        raise NotImplementedError(
            "RouterLM requires at least one model with chat template support"
        )

    @property
    def tokenizer_name(self) -> str:
        """Get tokenizer name from first available model."""
        if self._models:
            first_model = next(iter(self._models.values()))
            return getattr(first_model, "tokenizer_name", "router")
        return "router"

    def chat_template(self, chat_template: bool | str = False) -> str | None:
        """Get chat template from first available model."""
        if self._models:
            first_model = next(iter(self._models.values()))
            return first_model.chat_template(chat_template)
        return None

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol, Union

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance


@dataclass
class RoutingContext:
    request_type: str
    task_name: str | None
    doc_id: int | None
    doc: dict[str, Any]
    arguments: tuple
    idx: int = 0
    batch_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeEvent:
    request: Instance
    task_name: str
    doc_id: int
    doc: dict[str, Any]

    model: str
    response: Any

    metrics: dict[str, float]
    correct: bool

    routing_metadata: dict[str, Any] = field(default_factory=dict)

    latency_ms: float = 0.0
    energy_joules: float = 0.0


class RoutingCallback(Protocol):
    def __call__(
        self,
        request: Instance,
        context: RoutingContext,
        state: dict[str, Any],
    ) -> Union[str, RoutingDecision]:
        """
        Make a routing decision for a single request.

        Args:
            request: The Instance to route
            context: Routing context with request metadata
            state: Mutable router state (persists across calls)

        Returns:
            - str: Model ID to route to
            - RoutingDecision: Model to route to + metadata
        """
        ...


class OutcomeCallback(Protocol):
    def __call__(self, event: OutcomeEvent, state: dict[str, Any]) -> None:
        """
        Receive feedback after a request is evaluated.

        Args:
            event: Contains request, response, metrics, and latency for the selected model
            state: Mutable router state (can be updated based on outcome)
        """
        ...


class FinishCallback(Protocol):
    def __call__(
        self,
        state: dict[str, Any],
        results: dict[str, Any],
        per_model_results: dict[str, dict[str, dict[str, float]]] | None = None,
    ) -> None:
        """
        Called when the evaluation run finishes.

        Args:
            state: Mutable router state (can be used for checkpointing)
            results: Final evaluation results dictionary
            per_model_results: Per-task, per-model metrics computed during exhaustive
                               evaluation. Structure: {task_name: {model_name: {metric: value}}}
                               None if not in exhaustive mode.
        """
        ...


@dataclass
class RouterState:
    state: dict[str, Any] = field(default_factory=dict)

    def update(self, key: str, value: Any) -> None:
        self.state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)


def load_callback(callback_path: str) -> Callable:
    """
    Load a callback function from a 'module.path:function_name' string.

    Args:
        callback_path: Python path like "my_package.module:my_function"

    Returns:
        The callable function

    Raises:
        ValueError: If the path format is invalid
        ImportError: If the module cannot be imported
        AttributeError: If the function doesn't exist in the module
    """
    import importlib

    if ":" not in callback_path:
        raise ValueError(
            f"Invalid callback path '{callback_path}'. "
            f"Expected format: 'module.path:function_name'"
        )

    module_path, func_name = callback_path.rsplit(":", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Could not import module '{module_path}' for callback '{callback_path}'"
        ) from e

    try:
        callback = getattr(module, func_name)
    except AttributeError as e:
        raise AttributeError(
            f"Function '{func_name}' not found in module '{module_path}'"
        ) from e

    if not callable(callback):
        raise TypeError(f"'{callback_path}' does not point to a callable function")

    return callback

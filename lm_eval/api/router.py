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
    primary_model: str
    shadow_models: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeEvent:
    request: Instance
    task_name: str
    doc_id: int
    doc: dict[str, Any]

    primary_model: str
    shadow_models: list[str]

    all_responses: dict[str, Any]

    primary_metrics: dict[str, float]
    primary_correct: bool

    all_metrics: dict[str, dict[str, float]]
    all_correct: dict[str, bool]

    routing_metadata: dict[str, Any] = field(default_factory=dict)


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
            - str: Model ID to route to (no shadow evaluation)
            - RoutingDecision: Primary model + optional shadow models
        """
        ...


class OutcomeCallback(Protocol):
    def __call__(self, event: OutcomeEvent, state: dict[str, Any]) -> None:
        """
        Receive feedback after a request is evaluated.

        Args:
            event: Contains request, responses, and metrics for all models
            state: Mutable router state (can be updated based on outcome)
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

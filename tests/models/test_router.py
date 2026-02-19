"""Tests for RouterLM and routing infrastructure."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from lm_eval.api.instance import Instance
from lm_eval.api.router import (
    OutcomeEvent,
    RoutingContext,
    RoutingDecision,
    load_callback,
)
from lm_eval.models.router import RouterConfig, RouterLM


# ============== FIXTURES ==============


@pytest.fixture
def mock_lm():
    """Create a mock LM with all required methods."""
    lm = MagicMock()
    lm.loglikelihood.return_value = [(-1.5, True)]
    lm.generate_until.return_value = ["generated text"]
    lm.loglikelihood_rolling.return_value = [-2.5]
    lm.apply_chat_template.return_value = "chat template"
    lm.tokenizer_name = "mock-tokenizer"
    lm.chat_template.return_value = None
    return lm


@pytest.fixture
def sample_instance():
    """Create a sample Instance for testing."""
    return Instance(
        request_type="loglikelihood",
        doc={"question": "What is 2+2?", "answer": "4"},
        arguments=("What is 2+2?", "4"),
        idx=0,
        metadata=("test_task", 0, 1),
    )


@pytest.fixture
def sample_generate_instance():
    """Create a sample generate_until Instance."""
    return Instance(
        request_type="generate_until",
        doc={"prompt": "Hello"},
        arguments=("Hello", {"max_tokens": 10}),
        idx=0,
        metadata=("test_task", 0, 1),
    )


@pytest.fixture
def router_config_yaml(tmp_path: Path) -> str:
    """Create a temporary router config YAML file."""
    config = {
        "models": {
            "model_a": {"type": "dummy"},
            "model_b": {"type": "dummy"},
        },
        "routing": {
            "primary_model": "model_a",
            "shadow_mode": "none",
            "adaptive": False,
        },
    }
    config_path = tmp_path / "router_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)


@pytest.fixture
def router_with_mocks(mock_lm: MagicMock, router_config_yaml: str) -> RouterLM:
    """RouterLM instance with mocked underlying models."""
    with patch("lm_eval.models.router.get_model") as mock_get_model:
        mock_get_model.return_value.create_from_arg_obj.return_value = mock_lm
        router = RouterLM(config_path=router_config_yaml)
        return router


# ============== CONFIG TESTS ==============


class TestRouterConfig:
    def test_config_from_yaml(self, router_config_yaml: str):
        config = RouterConfig.from_yaml(router_config_yaml)
        assert "model_a" in config.models
        assert "model_b" in config.models
        assert config.routing["primary_model"] == "model_a"

    def test_config_missing_file(self):
        with pytest.raises(FileNotFoundError):
            RouterConfig.from_yaml("/nonexistent/path.yaml")

    def test_config_empty_routing(self, tmp_path: Path):
        config = {"models": {"model_a": {"type": "dummy"}}}
        config_path = tmp_path / "minimal_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        cfg = RouterConfig.from_yaml(str(config_path))
        assert cfg.routing == {}


# ============== ROUTING DECISION TESTS ==============


class TestRoutingDecision:
    def test_decision_with_string_primary(
        self, router_with_mocks: RouterLM, sample_instance: Instance
    ):
        def routing_callback(request, context, state):
            return "model_b"

        router_with_mocks._routing_callback = routing_callback
        decision = router_with_mocks._make_routing_decision(sample_instance)

        assert decision.primary_model == "model_b"
        assert decision.shadow_models == []

    def test_decision_with_routing_decision_object(
        self, router_with_mocks: RouterLM, sample_instance: Instance
    ):
        def routing_callback(request, context, state):
            return RoutingDecision(
                primary_model="model_a",
                shadow_models=["model_b"],
                metadata={"reason": "complex query"},
            )

        router_with_mocks._routing_callback = routing_callback
        decision = router_with_mocks._make_routing_decision(sample_instance)

        assert decision.primary_model == "model_a"
        assert decision.shadow_models == ["model_b"]
        assert decision.metadata["reason"] == "complex query"

    def test_decision_with_dict(
        self, router_with_mocks: RouterLM, sample_instance: Instance
    ):
        def routing_callback(request, context, state):
            return {"primary": "model_b", "shadow": ["model_a"]}

        router_with_mocks._routing_callback = routing_callback
        decision = router_with_mocks._make_routing_decision(sample_instance)

        assert decision.primary_model == "model_b"
        assert decision.shadow_models == ["model_a"]

    def test_no_callback_uses_default(
        self, router_with_mocks: RouterLM, sample_instance: Instance
    ):
        router_with_mocks._routing_callback = None
        decision = router_with_mocks._make_routing_decision(sample_instance)

        assert decision.primary_model == "model_a"


# ============== SHADOW MODE TESTS ==============


class TestShadowRouting:
    def test_shadow_mode_all(
        self, router_with_mocks: RouterLM, sample_instance: Instance
    ):
        router_with_mocks._shadow_mode = "all"
        decision = router_with_mocks._make_routing_decision(sample_instance)

        assert decision.primary_model == "model_a"
        assert "model_b" in decision.shadow_models

    def test_shadow_mode_none(
        self, router_with_mocks: RouterLM, sample_instance: Instance
    ):
        router_with_mocks._shadow_mode = "none"
        decision = router_with_mocks._make_routing_decision(sample_instance)

        assert len(decision.shadow_models) == 0

    def test_shadow_mode_sampled(
        self, router_with_mocks: RouterLM, sample_instance: Instance
    ):
        router_with_mocks._shadow_mode = "sampled"
        router_with_mocks._shadow_sample_rate = 1.0
        decision = router_with_mocks._make_routing_decision(sample_instance)

        assert decision.primary_model == "model_a"
        assert "model_b" in decision.shadow_models


# ============== PENDING DECISION TESTS ==============


class TestPendingDecisions:
    def test_pending_decision_stored_adaptive(
        self, router_with_mocks: RouterLM, sample_instance: Instance
    ):
        router_with_mocks._adaptive = True
        router_with_mocks.loglikelihood([sample_instance])

        key = (
            sample_instance.task_name,
            sample_instance.doc_id,
            sample_instance.idx,
        )
        assert key in router_with_mocks._pending_decisions

    def test_pending_decision_stored_with_shadow(
        self, router_with_mocks: RouterLM, sample_instance: Instance
    ):
        router_with_mocks._adaptive = False
        router_with_mocks._shadow_mode = "all"
        router_with_mocks.loglikelihood([sample_instance])

        key = (
            sample_instance.task_name,
            sample_instance.doc_id,
            sample_instance.idx,
        )
        assert key in router_with_mocks._pending_decisions

    def test_get_pending_decision_returns_and_clears(
        self, router_with_mocks: RouterLM, sample_instance: Instance
    ):
        router_with_mocks._adaptive = True
        router_with_mocks.loglikelihood([sample_instance])

        decision = router_with_mocks.get_pending_decision(
            sample_instance.task_name,
            sample_instance.doc_id,
            sample_instance.idx,
        )

        assert decision is not None
        assert decision.primary_model == "model_a"

        decision2 = router_with_mocks.get_pending_decision(
            sample_instance.task_name,
            sample_instance.doc_id,
            sample_instance.idx,
        )
        assert decision2 is None

    def test_get_pending_decision_missing_returns_none(
        self, router_with_mocks: RouterLM
    ):
        decision = router_with_mocks.get_pending_decision("nonexistent", 999, 999)
        assert decision is None


# ============== OUTCOME CALLBACK TESTS ==============


class TestOutcomeCallbacks:
    def test_outcome_callback_called(self, router_with_mocks: RouterLM):
        events_received = []

        def outcome_callback(event, state):
            events_received.append(event)

        router_with_mocks._outcome_callback = outcome_callback

        event = OutcomeEvent(
            request=MagicMock(),
            task_name="test_task",
            doc_id=0,
            doc={},
            primary_model="model_a",
            shadow_models=[],
            all_responses={"model_a": "response"},
            primary_metrics={"acc": 1.0},
            primary_correct=True,
            all_metrics={"model_a": {"acc": 1.0}},
            all_correct={"model_a": True},
        )

        router_with_mocks.on_outcome(event)

        assert len(events_received) == 1
        assert events_received[0].primary_correct is True

    def test_state_persists_across_callbacks(
        self, router_with_mocks: RouterLM, sample_instance: Instance
    ):
        call_count = {"count": 0}

        def routing_callback(request, context, state):
            state["count"] = state.get("count", 0) + 1
            call_count["count"] = state["count"]
            return "model_a"

        router_with_mocks._routing_callback = routing_callback

        for _ in range(3):
            router_with_mocks._make_routing_decision(sample_instance)

        assert call_count["count"] == 3

    def test_no_outcome_callback_no_error(self, router_with_mocks: RouterLM):
        router_with_mocks._outcome_callback = None

        event = OutcomeEvent(
            request=MagicMock(),
            task_name="test_task",
            doc_id=0,
            doc={},
            primary_model="model_a",
            shadow_models=[],
            all_responses={"model_a": "response"},
            primary_metrics={"acc": 1.0},
            primary_correct=True,
            all_metrics={"model_a": {"acc": 1.0}},
            all_correct={"model_a": True},
        )

        router_with_mocks.on_outcome(event)


# ============== CALLBACK LOADING TESTS ==============


class TestLoadCallback:
    def test_load_valid_callback(self):
        callback = load_callback("lm_eval.api.router:load_callback")
        assert callable(callback)

    def test_load_invalid_module(self):
        with pytest.raises(ImportError):
            load_callback("nonexistent.module:func")

    def test_load_missing_function(self):
        with pytest.raises(AttributeError):
            load_callback("lm_eval.api.router:nonexistent_func")

    def test_load_invalid_format(self):
        with pytest.raises(ValueError):
            load_callback("invalid_format_no_colon")


# ============== MODEL EXECUTION TESTS ==============


class TestModelExecution:
    def test_loglikelihood_routes_to_primary(
        self, router_with_mocks: RouterLM, sample_instance: Instance
    ):
        results = router_with_mocks.loglikelihood([sample_instance])

        assert len(results) == 1
        router_with_mocks._models["model_a"].loglikelihood.assert_called_once()

    def test_generate_until_routes_to_primary(
        self, router_with_mocks: RouterLM, sample_generate_instance: Instance
    ):
        results = router_with_mocks.generate_until([sample_generate_instance])

        assert len(results) == 1
        router_with_mocks._models["model_a"].generate_until.assert_called_once()

    def test_loglikelihood_rolling_routes_to_primary(self, router_with_mocks: RouterLM):
        instance = Instance(
            request_type="loglikelihood_rolling",
            doc={"text": "hello world"},
            arguments=("hello world",),
            idx=0,
            metadata=("test_task", 0, 1),
        )

        results = router_with_mocks.loglikelihood_rolling([instance])

        assert len(results) == 1
        router_with_mocks._models["model_a"].loglikelihood_rolling.assert_called_once()

    def test_shadow_models_executed(
        self, router_with_mocks: RouterLM, sample_instance: Instance
    ):
        router_with_mocks._shadow_mode = "all"
        router_with_mocks.loglikelihood([sample_instance])

        router_with_mocks._models["model_a"].loglikelihood.assert_called()
        router_with_mocks._models["model_b"].loglikelihood.assert_called()


# ============== ERROR HANDLING TESTS ==============


class TestErrorHandling:
    def test_invalid_primary_model(
        self, router_with_mocks: RouterLM, sample_instance: Instance
    ):
        def bad_callback(request, context, state):
            return "nonexistent_model"

        router_with_mocks._routing_callback = bad_callback

        with pytest.raises(ValueError, match="Primary model.*not found"):
            router_with_mocks._make_routing_decision(sample_instance)

    def test_invalid_shadow_model(
        self, router_with_mocks: RouterLM, sample_instance: Instance
    ):
        def bad_callback(request, context, state):
            return RoutingDecision(
                primary_model="model_a",
                shadow_models=["nonexistent_shadow"],
            )

        router_with_mocks._routing_callback = bad_callback

        with pytest.raises(ValueError, match="Shadow model.*not found"):
            router_with_mocks._make_routing_decision(sample_instance)


# ============== WEIGHTED RANDOM ROUTER INTEGRATION TESTS ==============


class TestWeightedRandomRouterIntegration:
    def test_router_can_be_loaded(self):
        callback = load_callback("examples.weighted_random_router:route")
        assert callable(callback)

    def test_router_route_returns_valid_model(self):
        from examples.weighted_random_router import WeightedRandomRouter

        state = {
            "weights": {"model_a": 0.5, "model_b": 0.5},
            "learning_rate": 0.1,
        }
        request = MagicMock()
        request.task_name = "test"
        request.doc_id = 0
        request.doc = {}
        context = RoutingContext(
            request_type="loglikelihood",
            task_name="test",
            doc_id=0,
            doc={},
            arguments=(),
        )

        result = WeightedRandomRouter.route(request, context, state)

        assert result in ("model_a", "model_b") or isinstance(result, RoutingDecision)

    def test_router_update_modifies_weights(self):
        from examples.weighted_random_router import WeightedRandomRouter

        state = {
            "weights": {"model_a": 0.5, "model_b": 0.5},
            "learning_rate": 0.5,
        }

        event = OutcomeEvent(
            request=MagicMock(),
            task_name="test",
            doc_id=0,
            doc={},
            primary_model="model_a",
            shadow_models=["model_b"],
            all_responses={},
            primary_metrics={},
            primary_correct=True,
            all_metrics={},
            all_correct={"model_a": True, "model_b": False},
        )

        original_weight_a = state["weights"]["model_a"]
        WeightedRandomRouter.update(event, state)

        assert state["weights"]["model_a"] != original_weight_a


# ============== PROPERTIES TESTS ==============


class TestRouterProperties:
    def test_adaptive_property(self, router_with_mocks: RouterLM):
        router_with_mocks._adaptive = True
        assert router_with_mocks.adaptive is True

        router_with_mocks._adaptive = False
        assert router_with_mocks.adaptive is False

    def test_primary_model_property(self, router_with_mocks: RouterLM):
        assert router_with_mocks.primary_model == "model_a"

    def test_models_property(self, router_with_mocks: RouterLM):
        models = router_with_mocks.models
        assert "model_a" in models
        assert "model_b" in models

    def test_state_property(self, router_with_mocks: RouterLM):
        router_with_mocks.update_state("test_key", "test_value")
        assert router_with_mocks.state["test_key"] == "test_value"

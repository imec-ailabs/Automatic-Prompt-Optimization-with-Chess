"""Offline Decisions payload, transport, and DSPy adapter contracts."""

import asyncio
import copy
import io
import json
import time
import urllib.request
from email.message import Message
from typing import Any
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import dspy
import pytest
from dspy.utils import DummyLM

from chess_self_improvement.dspy_program import (
    ChessMoveAnalysisSignature,
    ChessMoveSignature,
)
from chess_self_improvement.jev import (
    JEV_MODEL,
    JevAdapter,
    JevLM,
    JevTransport,
    UnsupportedJevSampling,
)

INPUTS = {
    "position_fen": "8/8/8/8/8/8/4K3/7k w - - 0 1",
    "side_to_move": "white",
    "legal_moves_uci": "e2e3, e2d3",
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", Mock(side_effect=AssertionError))


@pytest.fixture
def request_mock() -> Mock:
    return Mock(
        return_value={
            "model": JEV_MODEL,
            "answers": {
                "move": {
                    "type": "choice",
                    "choice": "e2e3",
                    "probabilities": {"e2e3": 0.7, "e2d3": 0.3},
                    "confidence": 0.4,
                }
            },
        }
    )


def test_payload_instructions_demos_and_target_isolation(request_mock: Mock) -> None:
    predictor = dspy.Predict(ChessMoveSignature)
    predictor.signature = predictor.signature.with_instructions("Prefer activity.")
    predictor.signature = predictor.signature.with_updated_fields(
        "move", prefix="Choice:", desc="Prefer central king moves."
    )
    predictor.demos = [dspy.Example(**INPUTS, move="e2d3", private="PRIVATE_DEMO")]
    with dspy.context(lm=JevLM(request=request_mock), adapter=JevAdapter()):
        result = predictor(**INPUTS, move="TARGET_SENTINEL", private="PRIVATE_INPUT")
    assert result.move == "e2e3"
    payload = request_mock.call_args.args[0]
    assert set(payload) == {"model", "state", "questions"}
    assert payload["model"] == JEV_MODEL
    assert payload["state"] == INPUTS
    question = payload["questions"]["move"]
    assert question["type"] == "choice"
    assert question["criteria"] == {"e2e3": None, "e2d3": None}
    for text in (
        "Prefer activity.",
        "Choice:",
        "Prefer central king moves.",
        '"move": "e2d3"',
    ):
        assert text in question["instructions"]
    assert "TARGET_SENTINEL" not in json.dumps(payload)
    assert "PRIVATE" not in json.dumps(payload)


@pytest.mark.parametrize(
    "signature", [ChessMoveAnalysisSignature, "position_fen -> move"]
)
def test_unsupported_signature_before_request(
    request_mock: Mock, signature: Any
) -> None:
    with (
        dspy.context(lm=JevLM(request=request_mock), adapter=JevAdapter()),
        pytest.raises(ValueError, match="single string move"),
    ):
        dspy.Predict(signature)(**INPUTS)
    request_mock.assert_not_called()


@pytest.mark.parametrize("moves", ["", "e2e3,", "e2e3,e2e3", "Ke3", "e9e3"])
def test_invalid_criteria_before_request(request_mock: Mock, moves: str) -> None:
    with (
        dspy.context(lm=JevLM(request=request_mock), adapter=JevAdapter()),
        pytest.raises(ValueError, match="distinct, nonempty UCI"),
    ):
        dspy.Predict(ChessMoveSignature)(**{**INPUTS, "legal_moves_uci": moves})
    request_mock.assert_not_called()


@pytest.mark.parametrize(
    "update,match",
    [
        ({"type": "score"}, "type choice"),
        ({"choice": "a1a2"}, "not in"),
        ({"choice": []}, "not in"),
        ({"probabilities": None}, "cover exactly"),
        ({"probabilities": {"e2e3": 1}}, "cover exactly"),
        ({"probabilities": {"e2e3": 0.6, "e2d3": 0.3}}, "sum to 1"),
        ({"confidence": float("nan")}, "finite"),
    ],
)
def test_invalid_answers(
    request_mock: Mock, update: dict[str, Any], match: str
) -> None:
    request_mock.return_value["answers"]["move"].update(update)
    with pytest.raises(ValueError, match=match):
        JevAdapter()(JevLM(request=request_mock), {}, ChessMoveSignature, [], INPUTS)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, "0.7", True])
def test_invalid_probabilities(request_mock: Mock, value: Any) -> None:
    request_mock.return_value["answers"]["move"]["probabilities"]["e2e3"] = value
    with pytest.raises(ValueError, match="finite"):
        JevAdapter()(JevLM(request=request_mock), {}, ChessMoveSignature, [], INPUTS)


@pytest.mark.parametrize(
    "response", [None, {}, {"model": "wrong"}, {"model": JEV_MODEL, "answers": {}}]
)
def test_invalid_response_envelope(request_mock: Mock, response: Any) -> None:
    request_mock.return_value = response
    with pytest.raises(ValueError):
        JevAdapter()(JevLM(request=request_mock), {}, ChessMoveSignature, [], INPUTS)


@pytest.mark.parametrize("second", [0.29, 0.31])
def test_rounding_and_selected_choice_preserved(
    request_mock: Mock, second: float
) -> None:
    answer = request_mock.return_value["answers"]["move"]
    answer.update(choice="e2d3", probabilities={"e2e3": 0.7, "e2d3": second})
    before = copy.deepcopy(request_mock.return_value)
    assert JevAdapter()(
        JevLM(request=request_mock), {}, ChessMoveSignature, [], INPUTS
    ) == [{"move": "e2d3"}]
    assert request_mock.return_value == before


@pytest.mark.parametrize(
    "config",
    [
        {"temperature": 0.7},
        {"rollout_id": 1},
        {"n": 2},
        {"top_p": 0.9},
        {"max_tokens": 20},
    ],
)
def test_sampling_rejected_before_request(
    request_mock: Mock, config: dict[str, Any]
) -> None:
    lm = JevLM(request=request_mock)
    with pytest.raises(UnsupportedJevSampling):
        lm.copy(**config)
    with (
        dspy.context(lm=lm, adapter=JevAdapter()),
        pytest.raises(UnsupportedJevSampling),
    ):
        dspy.Predict(ChessMoveSignature)(**INPUTS, config=config)
    request_mock.assert_not_called()


def test_copy_shares_request_but_not_history_or_secrets(request_mock: Mock) -> None:
    lm = JevLM(request=request_mock)
    lm.history.append({"private": "SECRET"})
    for cloned in (lm.copy(rollout_id=0), copy.deepcopy(lm)):
        assert cloned.request is request_mock
        assert cloned.history == []
        assert cloned.kwargs is not lm.kwargs
        assert cloned.callbacks is not lm.callbacks
    lm.kwargs["api_key"] = "SECRET"
    assert lm.dump_state() == {"model": JEV_MODEL, "temperature": 0, "n": 1}


def test_chat_meta_adapter_dispatch(request_mock: Mock) -> None:
    prompt = DummyLM([{"proposal": "Prefer forcing moves."}])
    adapter = JevAdapter(use_json_adapter_fallback=False)
    with dspy.context(lm=prompt, adapter=adapter):
        assert (
            dspy.Predict("task -> proposal")(task="Improve").proposal
            == "Prefer forcing moves."
        )
    assert "[[ ## proposal ## ]]" in prompt.history[0]["messages"][0]["content"]
    request_mock.assert_not_called()


def test_async_jev_rejected_without_io(request_mock: Mock) -> None:
    with pytest.raises(NotImplementedError, match="synchronous Predict"):
        asyncio.run(
            JevAdapter().acall(
                JevLM(request=request_mock), {}, ChessMoveSignature, [], INPUTS
            )
        )
    request_mock.assert_not_called()


@pytest.mark.parametrize("field", sorted(INPUTS))
def test_missing_input_rejected_before_request(request_mock: Mock, field: str) -> None:
    inputs = {key: value for key, value in INPUTS.items() if key != field}
    with pytest.raises(ValueError, match="all three chess inputs"):
        JevAdapter()(JevLM(request=request_mock), {}, ChessMoveSignature, [], inputs)
    request_mock.assert_not_called()


def test_incomplete_demo_rejected_before_request(request_mock: Mock) -> None:
    with pytest.raises(ValueError, match="labeled examples require"):
        JevAdapter()(
            JevLM(request=request_mock),
            {},
            ChessMoveSignature,
            [dspy.Example(**INPUTS)],
            INPUTS,
        )
    request_mock.assert_not_called()


def test_transport_payload_timeout_and_no_retained_credentials(
    monkeypatch: pytest.MonkeyPatch, request_mock: Mock
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "SECRET")
    http = Mock(return_value=io.BytesIO(json.dumps(request_mock.return_value).encode()))
    monkeypatch.setattr(urllib.request, "urlopen", http)
    transport = JevTransport(timeout_seconds=12, retries=0)
    lm = JevLM(request=transport)
    assert JevAdapter()(lm, {}, ChessMoveSignature, [], INPUTS) == [{"move": "e2e3"}]
    request = http.call_args.args[0]
    assert request.full_url == "https://openrouter.ai/api/alpha/decisions"
    assert request.method == "POST"
    assert request.get_header("Authorization") == "Bearer SECRET"
    assert request.get_header("Content-type") == "application/json"
    assert http.call_args.kwargs == {"timeout": 12}
    assert "SECRET" not in json.dumps(json.loads(request.data))
    assert vars(transport) == {"timeout_seconds": 12, "retries": 0}
    assert "SECRET" not in json.dumps(lm.dump_state())
    assert not lm.history


def test_transport_requires_environment_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        JevTransport()({})


@pytest.mark.parametrize("body", [b"not-json SECRET", b"[]"])
def test_transport_invalid_json_is_safe(
    monkeypatch: pytest.MonkeyPatch, body: bytes
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "SECRET")
    monkeypatch.setattr(urllib.request, "urlopen", Mock(return_value=io.BytesIO(body)))
    with pytest.raises(ValueError, match="JSON") as error:
        JevTransport()({})
    assert "SECRET" not in str(error.value)


@pytest.mark.parametrize("status,attempts", [(400, 1), (401, 1), (429, 2), (503, 2)])
def test_transport_transient_retries_and_redaction(
    monkeypatch: pytest.MonkeyPatch, status: int, attempts: int
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "SECRET")
    http = Mock(side_effect=HTTPError("url", status, "SECRET", Message(), None))
    sleep = Mock()
    monkeypatch.setattr(urllib.request, "urlopen", http)
    monkeypatch.setattr(time, "sleep", sleep)
    with pytest.raises(RuntimeError) as error:
        JevTransport(retries=1)({})
    assert "SECRET" not in str(error.value)
    assert http.call_count == attempts
    assert sleep.call_count == attempts - 1


def test_transport_timeout_retry_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "SECRET")
    http = Mock(side_effect=[URLError(TimeoutError()), io.BytesIO(b"{}")])
    monkeypatch.setattr(urllib.request, "urlopen", http)
    monkeypatch.setattr(time, "sleep", Mock())
    assert JevTransport(retries=1)({}) == {}
    assert http.call_count == 2

"""Standalone Decisions transport and DSPy adapter for independent positions."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import dspy

JEV_MODEL = "typesafe/jev-1.13-20260917"
_INPUT_FIELDS = {"position_fen", "side_to_move", "legal_moves_uci"}


class JevTransport:
    """POST Decisions requests without retaining credentials or raw responses."""

    def __init__(self, *, timeout_seconds: float = 30, retries: int = 3) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        if retries < 0:
            raise ValueError("retries must be nonnegative")
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Read the environment key at call time; retry only transient failures."""
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY must be set in the environment")
        request = urllib.request.Request(
            "https://openrouter.ai/api/alpha/decisions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds
                ) as response:
                    result = json.loads(response.read())
            except (urllib.error.URLError, TimeoutError) as exc:
                status = getattr(exc, "code", None)
                transient = (
                    status == 429
                    or (status is not None and status >= 500)
                    or isinstance(exc, TimeoutError)
                    or isinstance(getattr(exc, "reason", None), TimeoutError)
                )
                if isinstance(exc, urllib.error.HTTPError):
                    exc.close()
                if transient and attempt < self.retries:
                    time.sleep(0.25)
                    continue
                # Provider errors can echo headers; never persist their body/reason.
                raise RuntimeError(
                    f"Jev Decisions request failed (HTTP {status or 'unavailable'})"
                ) from None
            except ValueError:
                raise ValueError("Jev Decisions response must be valid JSON") from None
            if not isinstance(result, dict):
                raise ValueError("Jev Decisions response must be a JSON object")
            return result
        raise AssertionError("Unreachable retry state")


class UnsupportedJevSampling(ValueError):
    """Requested generative sampling cannot be represented by Decisions."""


def _validate_sampling(kwargs: dict[str, Any]) -> None:
    supported = {"temperature": (0,), "n": (1,), "rollout_id": (None, 0)}
    for key, value in kwargs.items():
        if key == "max_tokens" and value is None:
            continue  # BaseLM stores this unset generative default.
        if key not in supported or value not in supported[key]:
            raise UnsupportedJevSampling(
                f"Jev does not support {key}={value!r}; requires temperature=0, "
                "n=1, rollout_id=0 (or unset). SIMBA sampling is unsupported."
            )


class JevLM(dspy.BaseLM):  # type: ignore[misc]
    """Hold a pinned Decisions model and a shared request callable."""

    def __init__(
        self,
        *,
        request: Callable[[dict[str, Any]], dict[str, Any]],
        model: str = JEV_MODEL,
    ) -> None:
        super().__init__(model=model, temperature=0, n=1, cache=False)
        self.request = request

    def copy(self, **kwargs: Any) -> JevLM:
        """Copy DSPy configuration without copying the request's runtime state."""
        merged = {**self.kwargs, **kwargs}
        _validate_sampling(merged)
        cloned = copy.copy(self)
        cloned.kwargs = merged
        cloned.history = []
        cloned.callbacks = list(self.callbacks)
        return cloned

    def __deepcopy__(self, memo: dict[int, Any]) -> JevLM:
        """Keep the request shared when an optimizer deep-copies a program."""
        cloned = self.copy()
        memo[id(self)] = cloned
        return cloned

    def dump_state(self) -> dict[str, Any]:
        """Describe the LM without serializing its callback or runtime secrets."""
        return {"model": self.model, "temperature": 0, "n": 1}


class JevAdapter(dspy.ChatAdapter):  # type: ignore[misc]
    """Dispatch Jev decisions separately from generative chat prompt models."""

    def __call__(
        self,
        lm: dspy.BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[Any],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Build and validate one move decision, or use the normal chat adapter."""
        if not isinstance(lm, JevLM):
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)  # type: ignore[no-any-return]

        _validate_sampling({**lm.kwargs, **lm_kwargs})
        if (
            set(signature.input_fields) != _INPUT_FIELDS
            or list(signature.output_fields) != ["move"]
            or any(field.annotation is not str for field in signature.fields.values())
        ):
            raise ValueError(
                "Jev requires string position_fen, side_to_move, legal_moves_uci "
                "inputs and a single string move output; analysis/free outputs "
                "are unsupported."
            )
        if any(not isinstance(inputs.get(key), str) for key in _INPUT_FIELDS):
            raise ValueError("Jev requires all three chess inputs as strings")
        state = {key: inputs[key] for key in signature.input_fields}
        moves = [move.strip() for move in state["legal_moves_uci"].split(",")]
        if len(set(moves)) != len(moves) or any(
            re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", move) is None for move in moves
        ):
            raise ValueError(
                "legal_moves_uci must contain distinct, nonempty UCI moves"
            )

        guide = [signature.instructions, "Field guide:"]
        for name, field in signature.fields.items():
            metadata = field.json_schema_extra or {}
            guide.append(
                f"{name} | {metadata.get('prefix', '')} | {metadata.get('desc', '')}"
            )
        for index, demo in enumerate(demos, start=1):
            if any(
                key not in demo or not isinstance(demo[key], str)
                for key in signature.fields
            ):
                raise ValueError(
                    "Jev labeled examples require all chess inputs and move"
                )
            labeled = {key: demo[key] for key in signature.fields}
            guide.append(f"Labeled example {index}:\n{json.dumps(labeled)}")
        guide.append("Choose one UCI option for the current state, not an example.")
        criteria = dict.fromkeys(moves)
        payload = {
            "model": lm.model,
            "state": state,
            "questions": {
                "move": {
                    "type": "choice",
                    "instructions": "\n\n".join(guide),
                    "criteria": criteria,
                }
            },
        }
        response = lm.request(payload)
        if not isinstance(response, dict) or response.get("model") != lm.model:
            raise ValueError("Jev served model must match the pinned requested model")
        answers = response.get("answers")
        if not isinstance(answers, dict) or set(answers) != {"move"}:
            raise ValueError("Jev response must contain exactly the move answer")
        answer = answers["move"]
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError("Jev move answer must have type choice")
        selected = answer.get("choice")
        if not isinstance(selected, str) or selected not in criteria:
            raise ValueError("Jev selected choice is not in the legal move criteria")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(criteria):
            raise ValueError("Jev probabilities must cover exactly all legal options")
        values = list(probabilities.values())
        if "confidence" in answer:
            values.append(answer["confidence"])
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 1
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "Jev probabilities/confidence must be finite numbers in 0..1"
            )
        # Observed decimal rounding can produce 0.99; never renormalize the response.
        if abs(math.fsum(probabilities.values()) - 1) > 0.01 + 1e-12:
            raise ValueError("Jev probabilities must sum to 1 within 0.01")
        return [{"move": selected}]

    async def acall(
        self,
        lm: dspy.BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[dspy.Signature],
        demos: list[Any],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Preserve async chat dispatch without blocking on a synchronous request."""
        if isinstance(lm, JevLM):
            raise NotImplementedError(
                "JevAdapter supports synchronous Predict calls only"
            )
        return await super().acall(lm, lm_kwargs, signature, demos, inputs)  # type: ignore[no-any-return]

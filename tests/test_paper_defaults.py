"""Guard the optimizer defaults against drifting from the paper's Table 11."""

from typing import Any

import pytest

from chess_self_improvement.optimization.parameters import resolve_optimizer_params

# Appendix B.6, Table 11 of arXiv:2610.00416. Thread counts affect speed only.
PAPER_TABLE_11: dict[str, dict[str, Any]] = {
    "gepa": {
        "auto": "heavy",
        "candidate_selection_strategy": "pareto",
        "use_merge": True,
        "skip_perfect_score": True,
        "reflection_minibatch_size": 3,
        "num_threads": 16,
    },
    "mipro_v2": {
        "auto": "heavy",
        "minibatch": False,
        "max_bootstrapped_demos": 4,
        "max_labeled_demos": 4,
        "num_candidates": None,  # set by auto=heavy
        "num_trials": None,  # set by auto=heavy
        "num_threads": 16,
        "program_aware_proposer": True,
        "data_aware_proposer": True,
        "tip_aware_proposer": True,
        "fewshot_aware_proposer": True,
    },
    "simba": {
        "bsize": 32,
        "num_candidates": 6,
        "max_steps": 8,
        "max_demos": 4,
        "num_threads": 16,
    },
    "copro": {"breadth": 10, "depth": 3, "eval_num_threads": 16},
    "bootstrap_few_shot": {
        "max_bootstrapped_demos": 4,
        "max_labeled_demos": 16,
        "max_rounds": 1,
    },
    "bootstrap_random_search": {
        "max_bootstrapped_demos": 4,
        "max_labeled_demos": 16,
        "num_candidate_programs": 16,
        "num_threads": 16,
        "max_rounds": 1,
    },
}


@pytest.mark.parametrize("name", sorted(PAPER_TABLE_11))
def test_omitted_optimizer_block_matches_paper(name: str) -> None:
    defaults = resolve_optimizer_params(name).model_dump()
    for field, value in PAPER_TABLE_11[name].items():
        assert defaults[field] == value, f"{name}.{field}"


def test_gepa_can_opt_out_of_auto_budget() -> None:
    params = resolve_optimizer_params("gepa", {"auto": None})
    assert params.model_dump()["auto"] is None

# Paper Configurations

Each YAML in [`configs/`](../configs/) is a standalone `chess-bench evaluate`
input: no inheritance, no extra top-level fields. The eight baselines use the
plain Predict program (`use_analysis: false`); the two GEPA examples are
bounded optimization runs. This page gives the per-model settings and the
steps to run the paper's full protocol.

## Models And Settings

All generative targets use temperature 0 and one retry. `max_tokens` is the
per-call output cap; the optimization runs use the same per-model value as the
baseline evaluations. `extra` is forwarded to `dspy.LM`; every generative
config sets `drop_params: true`. The flags column gives
`extra.reasoning_effort` / `extra.include_reasoning`; "default" means neither
is set, so the provider's default reasoning applies.

| Paper model | `task_model.name` | Max tokens | Timeout (s) | Reasoning flags | Baseline YAML |
| --- | --- | ---: | ---: | --- | --- |
| Gemini 3.5 Flash Lite | `openrouter/google/gemini-3.5-flash-lite` | 1024 | 30 | `low` / `true` | `independent_positions_baseline.yaml` |
| GPT-4o Mini | `openrouter/openai/gpt-4o-mini` | 1024 | 30 | `none` / `false` | `independent_positions_gpt4o_mini_baseline.yaml` |
| Claude Haiku 4.5 | `openrouter/anthropic/claude-haiku-4.5` | 1024 | 30 | default | `independent_positions_claude_haiku_baseline.yaml` |
| Qwen3.8 27B | `openrouter/qwen/qwen3.8-27b` | 512 | 120 | default | `independent_positions_qwen_baseline.yaml` |
| GPT-5.6 Luna | `openrouter/openai/gpt-5.6-luna` | 4096 | 300 | `low` / `true` | `independent_positions_gpt56_luna_baseline.yaml` |
| DeepSeek V4 Pro 0813 | `openrouter/deepseek/deepseek-v4-pro-0813` | 512 | 120 | default | `independent_positions_deepseek_baseline.yaml` |
| Muse Spark 1.2 Contributor | `openrouter/meta/muse-spark-1.2-contributor` | 16384 | 600 | `low` / `true` | `independent_positions_muse_spark_baseline.yaml` |
| Jev 1.13 | `typesafe/jev-1.13-20260917` | — | 30 | — | `independent_positions_jev_baseline.yaml` |

Every optimization config uses the fixed meta-model for prompt proposal and
reflection:

```yaml
prompt_model:
  name: openrouter/google/gemini-3.5-flash
  temperature: 1
  max_tokens: 8192
  retries: 1
  timeout_seconds: 30
  extra: {drop_params: true}
```

BFS and BRS bootstrap demonstrations with the target model and do not propose
instructions; they still take the `prompt_model` block.

## Select A Configuration

1. Pick a baseline YAML and run it from the repository root; paths are relative
   to the working directory.
2. To optimize a different target, replace the whole `task_model` block of an
   optimizer config with the chosen baseline's block, unchanged. Keep the
   `prompt_model` block as is.
3. Choose `method`. An omitted `optimizer` block runs the optimizer with the
   paper's settings (Appendix B.6, Table 11: DSPy defaults with `auto: heavy`
   for GEPA and MIPROv2); check the cost of a full run first. The bounded GEPA
   examples set `optimizer.auto: null` so that the run-level
   `max_metric_calls: 50` applies. `max_metric_calls` controls GEPA only, and
   only when `auto` is null. With MIPROv2 `auto`, do not set
   `num_candidates` or `num_trials`.
4. Keep `use_analysis: false`; enabling the analysis field changes the program.
   `include_ground_truth_feedback` is false by default.

### Jev

Jev chooses among the supplied legal moves through OpenRouter's decision
endpoint (`https://openrouter.ai/api/alpha/decisions`) with the same
`OPENROUTER_API_KEY`, which needs access to that endpoint. It does not generate
text, so temperature, the output-token cap, and the JSON adapter do not apply
and its YAML omits them.

Jev supports `independent_positions` without the analysis field. SIMBA requires
temperature sampling and is not applicable; BFS and BRS run with
`optimizer.max_rounds: 1`. Any Jev optimization needs the explicit
`prompt_model` block above. `configs/independent_positions_jev_gepa.yaml` is
the bounded Jev example.

## Full Data And Three Seeds

For each condition, create three run configurations with `seed: 42`, `43`,
and `44` and distinct `output_dir` values, then run each with
`uv run chess-bench evaluate <config.yaml>`. Set `test_limit: 0` (or omit it)
for the full held-out set, and for optimization runs also `train_limit: 0` and
`validation_size: 140`: the 559 training puzzles then split into 419
training and 140 validation puzzles before expansion into positions, keeping
all positions of a puzzle together.

The full held-out set has 559 puzzles and 1,439 positions. `summary.json`
reports position-level `accuracy` and the paper's puzzle-level
`source_record_accuracy`, computed over the `source_records_complete` puzzles
whose positions were all evaluated (`test_limit` caps positions, so a bounded
run can stop partway through a puzzle). Each `results.json` row carries
`source_id`, `move_index`, and `solver_length`, named as in
`results/paper/independent_positions.csv.gz`, for your own grouping. Report the
mean and sample standard deviation of `source_record_accuracy` across the three
runs, as in paper Table 3.

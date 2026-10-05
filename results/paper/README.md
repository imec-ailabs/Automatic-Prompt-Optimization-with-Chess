# Paper Observation Data

Three CSV files hold the observations behind the plots and tables of
*Benchmarking Prompt Optimization of Large Language Models With Chess*.
`public_manifest.json.gz` accompanies them with the schema, coverage counts,
scientific identities, recorded settings, and file hashes. No credentials or
network access are needed to read them.

## Files

| Dataset | Rows | Columns | Contents |
| --- | ---: | ---: | --- |
| `independent_positions.csv.gz` | 513,745 | 53 | 482,065 independent positions and 31,680 engine-scored positions, with 420,231 exact move-only response strings |
| `prompts.csv` | 1,292 | 23 | 141 full compiled prompts, 331 annotated instruction units, 820 structured demonstrations |
| `rollouts.csv.gz` | 5,946 | 4 | 210 game terminals, 1,925 model attempts (28 illegal), 3,811 played plies |

Byte sizes and SHA-256 digests are in `manifest.files`. Sizes are compressed
downloads; stream the position file rather than loading it whole.

## Coverage

| Scope | Evaluations | Contents |
| --- | ---: | --- |
| Main independent-position panel | 165 | 237,435 positions: eight targets, 141 optimized evaluations and 24 baselines |
| Cross-model and native transfer | 168 | 241,752 positions |
| Appendix baseline cohort | 9 | 12,951 positions: Gemini 3.5 Flash, Gemini 3.5 Flash Lite, and GPT-4o Mini, three runs each |
| Independent-position regret | 33 | 31,680 engine-scored positions |
| Compiled prompts | 141 | 47 model/optimizer cells, seeds 42/43/44 (SIMBA is not applicable to Jev) |
| Game-play rollouts | 210 | seven conditions, three seeds, ten starting positions |

Position cohorts overlap: 10,073 observations belong to several evaluations
and are stored once; `evaluation_ids_json` lists every membership. Engine-scored
rows are separate from main-panel trials. `coverage.positions.complete` in the
manifest confirms that every expected position evaluation is present.

## Format And Joins

UTF-8 CSV with headers, LF records, and quoted multiline text. Parse `*_json`
columns with `json.loads`. Empty scalars mean unavailable or not applicable,
not zero or false; JSON `null` and absent keys are distinct. Boolean columns
mix `true`/`false` and `True`/`False`, so compare
case-insensitively (`value.lower() == "true"`). Preserve numeric precision.

The manifest has `schema_version=2`. `files` gives ordered CSV columns, row
counts, bytes, and hashes; `source_hashes` records the hashes of the inputs
the bundle was built from. `dictionaries` holds the joins:

| Dictionary | Join and meaning |
| --- | --- |
| `artifacts` | Position `artifact_ref` and `engine_artifact_ref` resolve to source artifact SHA-256 |
| `settings` | Position `settings_ref` resolves to program/config/input/reference-metadata hashes, `model_settings_ref`, and `model_settings_recorded` |
| `model_settings` | 33 distinct recorded settings payloads |
| `evaluations` | Every member of position `evaluation_ids_json`: model/optimizer/seed/repeat/cohort/donor identity, counts, status, cost ledgers, and `settings_refs` |
| `prompts` | `prompt_id`: model and meta-model IDs, method/seed, source program/text hashes, signature fields, dependency versions, and compile metadata |
| `annotation_protocols` | Prompt `annotation_protocol_ref`: the classifier model, settings, and system protocol used to annotate instruction units |
| `rollouts` | `rollout_id`: model/condition/seed/start/FEN/color, source hashes, program/config/signature hashes, engine version, and `settings_ref` |
| `rollout_settings` | Rollout `settings_ref`: recorded model, game, and engine settings |
| `rollout_sources` | Rollout `source_sha256`: source/report/config/input hashes |

IDs are opaque, deterministic for identical ordered inputs, and local to this
bundle. Some evaluations carry several provenance tuples in `settings`; keep
them all. The unique position key is `(record_kind, artifact_ref, source_id,
move_index)`. For analysis, expand the desired evaluation memberships and pair
observations by `(evaluation_id, source_id, move_index)`; a source ID alone is
not an observation.

Non-empty position and rollout program hashes join prompt
`source_program_sha256`; also match model, optimizer, seed, and donor context,
since unchanged programs share hashes. Signature hashes and text hashes
identify different objects, and empty hashes do not match. Requested and served
model IDs are retained; provider request and call IDs are not.

## Position Dictionary

| Columns | Meaning |
| --- | --- |
| `record_kind`, `cohorts_json`, `evaluation_ids_json` | Observation kind and all evaluation memberships; filter before aggregating |
| `model`, `optimizer`, `seed`, `evaluation_seed`, `repeat`, `donor_model`, `donor_seed` | Recorded identities; the per-membership manifest identity is authoritative where ledger seed, model seed, and repeat differ |
| `source_id`, `move_index`, `position_fen`, `expected_move`, `predicted_move` | Lichess source ID, zero-based solver-position index, board, reference UCI move, and recorded prediction (occasionally SAN) |
| `prediction_text_json` | Exact `{"text":"..."}` move-only response, whitespace included; blank means not retained, not an empty response |
| `correct`, `status`, `illegal`, `accepted`, `error`, `parser_mode`, `parser_version` | Result, failure category, and parser identity; legality or acceptance does not imply correctness |
| `rating`, `themes_json`, `solver_length`, `mate` | Source puzzle rating, themes, number of solver moves, and mate flag |
| `requested_model`, `served_model`, `finish_reason`, `retry_number` | Model identity and completion/retry evidence |
| `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `cached_prompt_tokens`, `cache_write_tokens`, `cost_usd`, `latency_ms`, `cache_status`, `elapsed_seconds` | Recorded usage, cost in USD, latency, and elapsed time; token counters can overlap, and a missing cost does not mean a free call |
| `engine`, `engine_depth`, `best_cp`, `post_cp`, `best_is_mate`, `post_is_mate`, `best_depth_reached`, `post_depth_reached`, `regret_cp` | Engine observations, actual depths reached, and regret, without rounding or imputation |
| `source_row`, `source_fields_json`, `artifact_ref`, `settings_ref` | Source row index, outcome/termination/attempt/repeat/engine-role facts, and hash/settings joins |

Legal moves and side to move are derivable from the FEN. Rating deviation,
popularity, play count, opening tags, and source-game IDs are not included;
FEN, themes, and source IDs are.

## Prompt Dictionary

`row_kind` is `prompt`, `unit`, or `demo`. The unique key is
`(prompt_id, row_kind, unit_id)`; the parent's `unit_id` is blank. The parent
`text` holds the full compiled prompt once. Child `text` is blank by design:
recover it as `parent_text[int(start):int(end)]`. Offsets count Unicode code
points, not bytes or tokens.

`label`, `raw_label`, `ambiguous`, `note`, and `annotation_status` hold the
instruction-unit annotations. Categories are format/chess/examples/persona/other
(raw `unclassified` maps to `other`); `annotation_status` marks whether a unit
was manually audited. Each stored demonstration is one `examples` unit. The
`selected_*` flags, `composition_optimizer`, `instruction_unchanged`,
`prompt_unchanged`, and `condition_no_update` record which prompts the paper's
figures and appendix use and whether an optimizer left the prompt unchanged
(baseline instruction and no stored demonstrations).

`instruction_characters`, `demo_characters`, and `prompt_characters` count the
instruction plus every stored demonstration field, including dataset metadata
that is never sent to the model (`puzzle_id`, `rating`, `themes`,
`position_index`). The paper's prompt-size table counts only what an optimizer
can change: the instruction plus each demonstration rendered with its four
signature fields. To reproduce it, compute per prompt
`instruction_characters + sum(2 + len("\n".join(f"{k}: {v}" for k, v in demo.items() if k in ("position_fen", "side_to_move", "legal_moves_uci", "move"))))`
over the prompt's `demo_json` rows, then average the three seeds. The fixed
signature and adapter scaffolding (686 characters) is identical for every
prompt and is excluded. Do not count parent and child content twice.

Signature fields, package versions, and compile summaries are in the manifest.
Static prompts are the compiled programs' instructions and demonstrations, not
the runtime adapter requests.

## Rollout Dictionary

The four columns are `rollout_id`, `event_type`, `event_index`, and
`observation_json`; the three scalar columns form the unique key. Indices are
zero-based within each event list; terminals use zero. Metadata joins are
listed above.

| Event | Nested observations |
| --- | --- |
| `terminal` | Outcome, termination, reward, elapsed time, final FEN, legal/illegal counts, metrics, and aggregate usage with coverage flags |
| `attempt` | Move/attempt indices, before/after boards, reference fields when recorded, accepted/illegal flags, the decision call, and parser resolution |
| `ply` | Ply number, actor, UCI move, before/after FEN, regret, and before/after engine `score_cp`/`mate_in`/`best_move_uci`/depth |

Calls keep the exact `raw_text`, requested and served models,
finish/retry/error/status fields, and usage; `raw_output` is present only when
it differs from `raw_text`. Accepted attempts and model plies overlap: do not
add their counts or equijoin their indices. Each rollout records its engine
version in the manifest (`rollouts` dictionary); the released rollouts used
Stockfish 18 at depth 20.

## Paper Map

Labels follow the manuscript's LaTeX labels. The manifest's `paper_sha256`
identifies the manuscript source the bundle was exported against; the
reference version is the arXiv paper (arXiv:2610.00416).

| Labels | Evidence and aggregation |
| --- | --- |
| `tab:puzzle-optim-results`, `fig:puzzle-algorithm-accuracy-violin` | Main-panel positions: all-correct source-record scores, then three-run means and sample SD; one-sided unadjusted Welch comparisons; no-update cells excluded |
| `tab:model-experiment-characteristics`, `fig:puzzle-cost-vs-performance` | Main baselines and selected optimizers: source-record accuracy with recorded usage, cost, and latency |
| `fig:puzzle-baseline-to-best-by-elo`, `fig:puzzle-baseline-to-best-by-theme`, `fig:puzzle-baseline-to-best-by-slice` | Selected baseline/optimizer source-record scores by rating and theme (themes overlap) |
| `fig:puzzle-baseline-to-best-by-move-index` | Position correctness by solver length and index |
| `fig:prompt-transferability` | Transfer recipient score minus main baseline; donor is the highest single-run score; three repeats |
| `fig:best-prompt-composition-by-model` | Character share of each unit category in the selected prompts; area is total prompt characters |
| `tab:puzzle-prompt-size`, `sec:prompts-size`, `sec:example-prompts` | Prompt sizes computed from the instruction and the four signature fields of each demonstration (see Prompt Dictionary), no-update flags, and appendix selections |
| `tab:playing-chess-transfer`, `tab:playing-chess-starting-positions` | Rollout events and metadata; ten distinct starting conditions |
| `tab:position-regret` | Engine-scored positions, 33 evaluations |
| `fig:puzzle-elo-focused`, `sub:puzzle-difficulty` | Appendix baseline cohort and main-panel baselines; the fitted Elo curves are analytic, not observations. The pooled correlations and odds ratios in `sub:puzzle-difficulty` are not exactly reproducible from this bundle |
| `tab:test-set-distribution` | The train/test CSVs under `data/`, each source counted once |
| `tab:openrouter-model-identifiers`, `tab:lm-config`, `tab:optimizer-hyperparameters`, `sec:hps` | Manifest identities and recorded configurations |
| `fig:overview`, `tab:algorithm-summary`, `sec:chess-notation-examples`, `sec:task-prompts` | Non-empirical manuscript material; no observation CSV |

A complete main or transfer evaluation has 559 source records and 1,439 solver
positions. For the paper score, require every position of a source record to
be correct, divide by 559, then aggregate across runs. Position-level accuracy
(the local runner's `accuracy`) has a different denominator. Each condition
has three runs, identified by `seed` (and by `repeat` for transfer
evaluations). Transfer donors are chosen by highest single-run score, which can differ from the best-mean selection
(GPT-4o Mini donates COPRO; Luna is not a donor); unchanged-prompt observations
are kept.

Independent-position regret uses 240 long sources (960 positions) per
evaluation at depth 20, mate mapped to 100,000 cp and illegal moves to
200,000 cp. The engine version is recorded in the `engine` column; to score new regret or
run new rollouts, use Stockfish 18 at depth 20 for both, as described in the
paper. Rollout means pool accepted model-ply regrets
within each seed, then take mean and sample SD across seeds; medians pool moves
across seeds. Illegal rates count illegal attempts over all attempts, not
timeouts; checkmate losses are terminal checkmates with `terminal_outcome=loss`.
Do not mix regret scores with rollout engine scores.

Compile cost ledgers are available for 182 evaluations. Total and component
costs, row-level usage, and prompt summaries overlap; do not sum them together.

## Integrity Check

```python
import csv, gzip, hashlib, json
from pathlib import Path

root = Path("results/paper")
with gzip.open(root / "public_manifest.json.gz", "rt") as stream:
    manifest = json.load(stream)
for name, entry in manifest["files"].items():
    with (root / name).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == entry["sha256"]
with gzip.open(root / "independent_positions.csv.gz", "rt", newline="") as stream:
    for row in csv.DictReader(stream):
        settings = manifest["dictionaries"]["settings"][row["settings_ref"]]
        # Select a record kind and evaluation cohort before aggregating.
```

## Provenance

The bundle was exported from the experiment logs. The manifest's
`source_hashes` and per-artifact SHA-256 values identify those inputs. Gzip
output uses a zero mtime, no filename, and level 9. The Integrity Check snippet
above verifies the published files against the manifest.

## Licensing

Source chess records derive from the [Lichess database](https://database.lichess.org/#puzzles),
released under CC0 1.0 Universal. The repository's [Apache-2.0 license](../../LICENSE)
covers the code. The data files in this directory are released under the
[Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/);
please cite the paper when you use them. Model responses are included as research
data, subject to the terms of the respective model providers; treat text fields
as data (disable formula interpretation when opening CSVs in a spreadsheet).
Stockfish is GPL-3.0 software and is not bundled.

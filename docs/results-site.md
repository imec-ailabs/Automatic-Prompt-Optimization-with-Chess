# Interactive paper results

The standalone `site/` compares mean inference cost with puzzle accuracy.
It has no runtime dependencies, third-party scripts, or API calls. Paths are
relative and work at a project GitHub Pages URL.

## Preview and regenerate

From the repository root:

```bash
uv run --no-project python scripts/build_results_site.py
uv run --no-project python -m http.server 8000 --directory site --bind 127.0.0.1
```

Visit <http://127.0.0.1:8000>. Use HTTP rather than opening the HTML directly,
because the browser loads modules and `results.json`.

## Figure definition

- **Cost:** recorded evaluation USD for the full held-out test set, 559 puzzles
  expanded into 1,439 solver positions. No compilation cost or additional baseline
  cost is included. Input length affects this cost, but output tokens, model
  pricing and recorded cache effects also matter; it is not a pure length metric.
- **Aggregation:** one point per model–method condition. Both axes average the
  same retained runs. `pointsFor` excludes unchanged optimizer runs before
  aggregation and retains baselines. The 21 exclusions leave 144 runs across
  51 conditions. Counts are shown; SD is unavailable for n=1. The paper's
  Table 3 averages all runs, including unchanged ones.
- **Scale:** square root only, with USD tick labels. Equal visual distances are
  neither equal cost differences nor equal ratios.
- **Navigation:** box zoom by dragging empty plot space, and a discreet Reset
  view control at the upper right. No pan, wheel zoom, scale selector or
  individual-run plot mode. Artifact seed selection only changes the inspector.
- **Frontier:** calculated from all filtered conditions, independent of viewport.
  Error bars show ±1 sample SD and do not enter dominance tests.

All six applicable Jev conditions have evaluation costs and appear in this view.

## Data and provenance

The exporter verifies the CSV hashes, streams main-panel memberships, reconstructs
all-correct puzzle scores, and checks counts against the release manifest.
Evaluation ledgers take priority over complete position-cost sums; overlapping
evidence is never added. Missing costs remain null and are listed separately.

All 165 source runs and 141 optimized-run artifacts remain in `results.json`.
Model/method/seed joins must be unique. Prompt text is copied exactly from parent
CSV rows; baseline text comes from `ChessMoveSignature`. Baseline/compilation
metadata retained in JSON is not used as the displayed inference cost.

The inspector offers all retained seed artifacts. Training indices preserve the
original trace semantics; missing histories are labeled rather than inferred from
configured limits or model-call counts. Text is rendered using `textContent`.

## Checks

```bash
uv run pytest tests/test_results_site.py
node --test tests/results_site.test.mjs
uv run ruff check scripts/build_results_site.py tests/test_results_site.py
uv run mypy scripts/build_results_site.py tests/test_results_site.py
```

Use Node 22+. Browser smoke checks should cover model/method filters, exact prompt
inspection, seed selection, box zoom/reset, empty filters and a phone viewport.

## Publish

For a public `github.io` URL, deploy from **github.com**, not the Enterprise
development repository. The public destination named in this release's README is
`imec-ailabs/Automatic-Prompt-Optimization-with-Chess`. Once deployed there, the
expected URL (unless a custom domain is configured) is:

<https://imec-ailabs.github.io/Automatic-Prompt-Optimization-with-Chess/>

1. Include these changes in the public release repository's `main` branch.
2. In **Settings → Pages → Build and deployment**, choose **GitHub Actions**.
3. Run **Publish paper results** from Actions. Subsequent relevant pushes trigger
   it automatically; the deployment job reports the Pages URL.

The workflow rebuilds the dataset and uploads only `site/`. Enterprise support
for Pages actions and runner labels depends on the installation. Branch-based
Pages can instead serve the contents of `site/` from the designated branch root.
The page's code/data links already target the public destination above. Merging
the Enterprise PR alone does not publish to github.com or create a github.io URL.

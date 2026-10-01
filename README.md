# Benchmarking Large Language Model Prompt Optimization With Chess

This repository presents paper results on automatic prompt optimization for
large language models using chess. The study compares a baseline with six
prompt optimizers: GEPA, MIPROv2, SIMBA, COPRO, BootstrapFewShot, and
BootstrapFewShotWithRandomSearch.

Code and datasets are not yet included in this repository.

## Paper Results

**Table 2.** Accuracy (%) over **559 source records**, each counted correct only
when **all its solver positions** are correct. Values are mean with sample
standard deviation over three runs, shown in compact type. This is not
position-level accuracy.

| <sub>Model</sub> | <sub>Baseline</sub> | <sub>BFS</sub> | <sub>BRS</sub> | <sub>COPRO</sub> | <sub>GEPA</sub> | <sub>MIPROv2</sub> | <sub>SIMBA</sub> |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <sub>GPT-4o Mini</sub> | <sub>5.84 &plusmn;0.37</sub> | <sub>5.01 &plusmn;0.31</sub> | <sub>4.95 &plusmn;0.99</sub> | <sub>5.84 &plusmn;0.55</sub> | <sub>5.66 &plusmn;0.52</sub> | <sub>4.95 &plusmn;0.81</sub> | <sub><strong>5.90 &plusmn;0.36</strong></sub> |
| <sub>Jev 1.13<sup>&dagger;</sup></sub> | <sub>7.27 &plusmn;0.10</sub> | <sub>7.87 &plusmn;0.47</sub> | <sub>8.77 &plusmn;0.31<sup>*</sup></sub> | <sub>7.69 &plusmn;0.18</sub> | <sub><strong>9.30 &plusmn;0.47</strong><sup>*</sup></sub> | <sub>8.59 &plusmn;0.65</sub> | <sub>N.A.</sub> |
| <sub>Qwen3.8 27B<sup>&dagger;</sup></sub> | <sub>8.47 &plusmn;0.37</sub> | <sub>9.48 &plusmn;0.36<sup>*</sup></sub> | <sub>9.78 &plusmn;0.41<sup>*</sup></sub> | <sub>9.12 &plusmn;0.31<sup>*</sup></sub> | <sub><strong>10.44 &plusmn;0.90</strong><sup>*</sup></sub> | <sub>8.23 &plusmn;0.31</sub> | <sub>9.36 &plusmn;0.45<sup>*</sup></sub> |
| <sub>Claude Haiku 4.5</sub> | <sub>11.63 &plusmn;0.00</sub> | <sub>11.09 &plusmn;0.18</sub> | <sub>11.21 &plusmn;1.02</sub> | <sub>N.U.</sub> | <sub><strong>12.28 &plusmn;1.22</strong></sub> | <sub>10.61 &plusmn;1.19</sub> | <sub>11.39 &plusmn;0.63</sub> |
| <sub>DeepSeek V4 Pro 0813<sup>&dagger;</sup></sub> | <sub>14.49 &plusmn;0.18</sub> | <sub>14.55 &plusmn;0.63</sub> | <sub>14.85 &plusmn;1.00</sub> | <sub>14.79 &plusmn;0.27</sub> | <sub><strong>16.28 &plusmn;0.31</strong><sup>*</sup></sub> | <sub>14.49 &plusmn;0.62</sub> | <sub>15.03 &plusmn;0.82</sub> |
| <sub>Gemini 3.5 Flash Lite<sup>&dagger;</sup></sub> | <sub>26.30 &plusmn;0.62</sub> | <sub>32.38 &plusmn;0.47<sup>*</sup></sub> | <sub>33.21 &plusmn;1.15<sup>*</sup></sub> | <sub>29.10 &plusmn;3.46</sub> | <sub>27.01 &plusmn;0.18</sub> | <sub>32.98 &plusmn;1.72<sup>*</sup></sub> | <sub><strong>33.87 &plusmn;1.03</strong><sup>*</sup></sub> |
| <sub>GPT-5.6 Luna</sub> | <sub><strong>31.72 &plusmn;0.37</strong></sub> | <sub>23.20 &plusmn;1.79</sub> | <sub>N.U.</sub> | <sub>30.95 &plusmn;1.46</sub> | <sub>31.48 &plusmn;1.17</sub> | <sub>30.05 &plusmn;0.78</sub> | <sub>N.U.</sub> |
| <sub>Muse Spark 1.2 Contributor</sub> | <sub>31.90 &plusmn;1.49</sub> | <sub>28.92 &plusmn;1.97</sub> | <sub>30.59 &plusmn;0.36</sub> | <sub>N.U.</sub> | <sub>32.20 &plusmn;1.40</sub> | <sub>29.93 &plusmn;1.14</sub> | <sub><strong>32.74 &plusmn;1.09</strong></sub> |

Bold preserves the manuscript's best-mean markings, including Luna's baseline.
<sup>*</sup> Improvement over the three-run baseline under an **unadjusted,
one-sided Welch test**, p &lt; 0.05; <sup>&dagger;</sup> at least one starred
optimizer. N.U.: none of the three optimizer seeds updated the original prompt.
N.A.: method not applicable. The manuscript's gray shading for below-baseline
means is omitted for portable GitHub rendering; all values and markers are retained.

<p>
  <img src="figures/puzzle-cost-vs-performance.png" width="48%" alt="Evaluation cost versus source-record accuracy, from baseline circles to best-optimizer stars">
  <img src="figures/puzzle-algorithm-accuracy-violin.png" width="48%" alt="Per-seed accuracy change from matched baseline, by optimizer">
</p>

**Figure 1 panels.** Left: mean evaluation cost versus mean source-record
accuracy across three seeds, from baseline (open circle) to best algorithm
(star). Right: per-seed accuracy changes from the matched baseline by optimizer.
Figures rendered from the original paper PDFs.

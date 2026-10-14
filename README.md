# Saina × Decision Index

Run any [Saina](https://saina.run) model through the [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
benchmark on your own GPU, with the official kit and scorers, and get the same files we publish.

- **Model:** pulled from the Hugging Face Hub at an exact commit (default
  [`run-saina/saina-helm-0.8b`](https://huggingface.co/run-saina/saina-helm-0.8b)).
- **Benchmark:** the official [Decision Index kit](https://github.com/apolinario/decision-index), pinned to the
  edition 0.3 release commit. Its suite build, hash checks, runner and scorers are used unchanged.
- **Adapter:** [`saina.jev`](https://pypi.org/project/saina/) from the `saina` SDK, the same code that serves
  `/v1/systemone` on api.saina.run. Every benchmark gets one fixed rendering; no labels, benchmark names or
  prompt tuning reach the model.

## Results

| Model | Revision | Public index (0.3) | Run |
| --- | --- | --- | --- |
| [Saina Helm 2 0.8B](https://huggingface.co/run-saina/saina-helm-2-0.8b) | `15745e48` (current) | **31.51** | [run-saina/saina-decision-index-results](https://huggingface.co/datasets/run-saina/saina-decision-index-results) (`runs/15745e48…/`) |
| [Saina Helm 2 0.8B](https://huggingface.co/run-saina/saina-helm-2-0.8b) | `06f2fec2` | 30.47 | [run-saina/saina-decision-index-results](https://huggingface.co/datasets/run-saina/saina-decision-index-results) (`runs/06f2fec2…/`) |
| [Saina Helm 2 0.8B](https://huggingface.co/run-saina/saina-helm-2-0.8b) | `9903c487` | 30.17 | [run-saina/saina-decision-index-results](https://huggingface.co/datasets/run-saina/saina-decision-index-results) (`runs/9903c487…/`) |
| [Saina Helm 2 0.8B](https://huggingface.co/run-saina/saina-helm-2-0.8b) | `76470f9b` | 29.49 | [run-saina/saina-decision-index-results](https://huggingface.co/datasets/run-saina/saina-decision-index-results) (`runs/76470f9b…/`) |
| [Saina Helm 2 0.8B](https://huggingface.co/run-saina/saina-helm-2-0.8b) | `63dfcafd` | 28.42 | [run-saina/saina-decision-index-results](https://huggingface.co/datasets/run-saina/saina-decision-index-results) (root) |

Each run used this kit on one RTX 3090: all 43 benchmarks, 140,178 scored requests, none failed.
By area for `15745e48`: tools 58.7, retrieval 43.7, language 31.6, knowledge 11.4, arts 9.4. This is the public part of
the index, self-run; among models under 1B on the board that is fourth (leaderboard data of 2026-10-14). The board's
official Full score, which adds private tests, is not yet available for `15745e48`. Per-benchmark results
are on the [model card](https://huggingface.co/run-saina/saina-helm-2-0.8b#evaluation).

To check it, re-score the published outputs (after building the suite, below):

```sh
hf download run-saina/saina-decision-index-results runs/15745e480c90b471258f6589deedccb79cf054f9/results.jsonl --repo-type dataset --local-dir results
python -m decision_index score --results results/runs/15745e480c90b471258f6589deedccb79cf054f9/results.jsonl --suite-dir suite-0.3 --out rescore
```

## Quickstart

Needs Python 3.11+ and a CUDA GPU (the 0.8B model fits comfortably in 8 GB).

```sh
git clone https://github.com/run-saina/saina-decision-index && cd saina-decision-index
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Build the suite once. It is not redistributable, so everyone rebuilds it from its pinned public sources
(about 7 GB of downloads and 17 GB of working space). Log in to the Hub and accept the
[HLE](https://huggingface.co/datasets/cais/hle) terms first:

```sh
hf auth login
scripts/build-suite.sh
```

The build ends with `suite verify`, which refuses any file whose hash differs from the board's manifest.

Run and score:

```sh
LIMIT=200 scripts/run.sh   # smoke test
scripts/run.sh             # full suite: 140,178 requests
```

Results land in `runs/<model>/<revision>/`:

| File | Contents |
| --- | --- |
| `results.jsonl` | Every request with the model's answer and full probability distribution, latency and status |
| `benchmark-summary.json` | Native metric per benchmark, with answered / unsupported / error counts |
| `index.json` | Chance-corrected, coverage-adjusted score per benchmark, per area, and the public Decision Index |
| `scores.json` | Headline scores and completeness |
| `environment.json` | GPU, library versions, model repo and revision, adapter and engine hashes, this repo's commit |

Interrupted runs resume where they stopped; rows that errored are retried.

## Other models and revisions

The engine loads any Saina checkpoint on the Hub that ships a `head-config.json`. Helm 2 models need
`saina` 0.3 or newer:

```sh
MODEL=run-saina/<model> REVISION=<40-char commit> scripts/run.sh
```

A revision must be a full commit hash, never a branch name, so every result names exactly the weights it
measured.

## Publishing a run

Upload a run to a Hub dataset you own, under the same `runs/<model>/<revision>/` layout:

```sh
UPLOAD=<you>/<dataset> scripts/run.sh
```

To check someone else's run, run the same model and revision and compare `index.json`, or re-score their
`results.jsonl` directly with `python -m decision_index score --results <path>/results.jsonl`.

## Protocol

These follow the kit's own rules:

- **No truncation.** A request that does not fit the model's context window, has more options than the
  model's answer head supports, or falls outside the `/v1/systemone` request contract is recorded as
  `unsupported` and scored as wrong. Nothing is cut or dropped to make it fit.
- **Questions.** Helm 2 models answer every question in a request from one pass over the shared state;
  earlier Helm models score each question with its own pass. `noul` questions become a two-option yes/no
  choice with their criteria as descriptions.
- **Public part only.** The kit computes the public index. The board's Full score also includes private
  tests run by its maintainers on submitted models.

## Using the engine directly

```sh
python -m decision_index run --engine saina_decision_index:SainaEngine \
    --option model=run-saina/saina-helm-0.8b \
    --option 'revision="b85231b8ca2f4f66cfd2e177d59635515474c736"' \
    --rows sample-100.jsonl.gz --out runs/sample
```

Options: `model`, `revision` (required), `device` (default `cuda`), `max_length` (default: the checkpoint's own
context limit), `conv_kernel` (see below).

## Faster kernels

Saina Helm is built on Qwen3.5, whose linear-attention layers run much faster with
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) and a fused causal convolution:

```sh
pip install -e ".[fast]"
CONV_KERNEL=20b2508ad12ae40260291539bf45183000451850 scripts/run.sh
```

`flash-linear-attention` is picked up automatically once installed. `CONV_KERNEL` loads the prebuilt causal
convolution from [`kernels-community/mamba-ssm`](https://huggingface.co/kernels-community/mamba-ssm) at that
commit, so no CUDA compiler is needed. Answers match the reference path within bfloat16 rounding; the kernel
used is recorded in `environment.json`.

## Tests

```sh
pip install -e ".[dev]" && pytest
```

The tests use a stand-in scorer and need no GPU or download.

## License

Apache-2.0. The Decision Index kit and each benchmark's sources keep their own licences.

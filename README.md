# Pen-DHRL: Deep Hierarchical RL for Generalizing Penetration-Testing Agents

Artifact for the AISec 2026 submission. This anonymized repository provides supporting evidence for the work presented in the paper, including the simulator, the hierarchical reinforcement learning agent (Pen-DHRL), dynamic corporate-network scenarios, and selected training and evaluation scripts.

The repository covers most of the components and experiments described in the paper; however, it is not intended to serve as a complete, fully reproducible package. Some implementation details, configurations, auxiliary files, or experimental components may be omitted from this review snapshot.

> **Anonymity note.** This repository was prepared specifically for double-blind review and contains only the materials considered relevant for evaluating and verifying the reported work.


---

## 1. Environment & installation

- **OS:** Linux (tested on Ubuntu-family distributions).
- **Python:** 3.10 (the agent code and pinned dependencies target CPython 3.10).
- **Hardware:** training is CPU-bound; no GPU is required.

```bash
# from the repository root
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# install the simulator package (declared in pyproject.toml)
pip install -e .

# agent-side dependencies
pip install torch torch-geometric numpy wandb
```

The environment package (`nasimemu`) is defined in [pyproject.toml](pyproject.toml).
`torch`/`torch-geometric` are used only by the agent code under
[`NASimEmu-agents/`](NASimEmu-agents/) and are installed separately so the
simulator can be used standalone.

---

## 2. Repository layout

| Path | Contents |
| --- | --- |
| [`src/nasimemu/`](src/nasimemu/) | Gym-based network attack **simulator** (based on Network Attack Simulator). |
| [`NASimEmu-agents/`](NASimEmu-agents/) | RL training/eval code; the **Pen-DHRL** agent lives in [`nasim_problem/nasim_net_base_hrl.py`](NASimEmu-agents/nasim_problem/nasim_net_base_hrl.py) (`NASimNetDHRL`). |
| [`scenarios/`](scenarios/) | Dynamic scenario definitions (see mapping below). |
| [`scenarios/regimes/`](scenarios/regimes/) | Fixed-regime ablation scenarios (`ids_off`, `defended`). |
| [`scripts/generate_regime_scenarios.py`](scripts/generate_regime_scenarios.py) | Regenerates the fixed-regime scenarios from the curriculum definitions. |

---

## 3. Scenario / configuration mapping

All training/eval commands run from inside `NASimEmu-agents/` and reference
scenarios via `../scenarios/...`. Multiple scenarios are joined with `:`.

| Scenario file | Role |
| --- | --- |
| `corp_100hosts_dynamic.v2.yaml` | Base 100-host dynamic corporate network (primary train/test scenario). |
| `corp_100hosts_dynamic_varA.v2.yaml`, `corp_100hosts_dynamic_varB.v2.yaml` | Curriculum variants; joined with the base for multi-scenario training. |
| `corp_100hosts_dynamic_bridge.v2.yaml` | Bridged-topology variant. |
| `corp_100hosts_dynamic_test.v2.yaml` | Held-out test scenario (novel-scenario generalization). |
| `corp.v2.yaml`, `uni.v2.yaml` | Smaller dynamic reference scenarios. |
| `regimes/ids_off/*.v2.yaml` | Ablation: defenses/IDS disabled (curriculum first stage held fixed). |
| `regimes/defended/*.v2.yaml` | Ablation: full defenses on (curriculum final stage held fixed). |

Regenerate the regime scenarios with:

```bash
python scripts/generate_regime_scenarios.py
```

---

## 4. Training

Run from inside `NASimEmu-agents/`:

```bash
cd NASimEmu-agents
```

Constraints on parallelism:
- `-batch` must be an exact multiple of `-cpus`.
- `-cpus` must evenly divide 64 (the eval pass uses a fixed batch of 64).
  Safe values: 1, 2, 4, 8, 16, 32, 64.

### 4a. Quick smoke run (~15–25 min, CPU)

Confirms the setup end-to-end (100 train-steps).

```bash
python main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  -device cpu -cpus 8 -batch 16 \
  -epoch 10 -max_epochs 10 \
  --no_debug \
  -net_class NASimNetDHRL \
  -use_a_t \
  -episode_step_limit 200 \
  -observation_format graph_v2 \
  -lr 0.0007 -alpha_h 0.02
```

### 4b. Full training run (reported results)

20,000 train-steps with the curriculum learning-rate / entropy schedule.

```bash
python main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic.v2.yaml \
  -device cpu -cpus 32 -batch 128 \
  -epoch 100 -max_epochs 200 \
  --no_debug \
  -net_class NASimNetDHRL \
  -force_continue_epochs 0 \
  -use_a_t \
  -episode_step_limit 400 \
  -observation_format graph_v2 \
  -lr 0.0007 -alpha_h 0.02 \
  --sched_lr_rate 10000 --sched_lr_factor 0.8 --sched_lr_min 0.0003 \
  --sched_alpha_h_rate 15000 --sched_alpha_h_factor 0.5 --sched_alpha_h_min 0.005 \
  -seed 1
```

Checkpoints are written to `wandb/offline-run-*/files/model.pt` (every epoch)
and `model_best.pt` (best `captured_avg` on the held-out test split). Pass
`-seed <n>` for a reproducible run; omit it for an exploratory run.

---

## 5. Evaluation

Evaluation reuses `main.py` with `--eval` and `-load_model`. The
architecture-affecting flags (`-net_class`, `-use_a_t`, `-episode_step_limit`,
`-observation_format`) **must match the checkpoint's training config**; `-cpus`
/ `-batch` / `-epoch` need not.

The `RUN_DIR=$(...)` line below is a real command that selects the most recent
run directory — copy and run the whole block together.

```bash
RUN_DIR=$(ls -t wandb/ | grep offline-run | head -1)

python main.py \
  ../scenarios/corp_100hosts_dynamic.v2.yaml:../scenarios/corp_100hosts_dynamic_varA.v2.yaml:../scenarios/corp_100hosts_dynamic_varB.v2.yaml \
  --test_scenario ../scenarios/corp_100hosts_dynamic_test.v2.yaml \
  --eval \
  -load_model wandb/$RUN_DIR/files/model_best.pt \
  -net_class NASimNetDHRL \
  -use_a_t \
  -episode_step_limit 400 \
  -observation_format graph_v2
```

Evaluate on `--test_scenario ../scenarios/corp_100hosts_dynamic_test.v2.yaml`
to measure generalization to the held-out scenario, or on the regime scenarios
under `../scenarios/regimes/{ids_off,defended}/` for the defense ablations.

---

## 6. Expected runtime & hardware

| Run | Config | Hardware | Wall-clock |
| --- | --- | --- | --- |
| Smoke (4a) | `-cpus 8 -batch 16`, 100 steps | 12-thread laptop CPU | ~15–25 min |
| Full (4b) | `-cpus 16 -batch 128`, 20k steps | 12-thread laptop CPU | ~52 h (measured) |
| Full (4b) | `-cpus 32 -batch 128`, 20k steps | 32-thread workstation CPU | ~18–21 h (estimated) |

The workload is CPU-bound; a GPU does not accelerate it. Scale `-cpus`/`-batch`
to the available core count while respecting the divisibility constraints in §4.

---

## 7. Reproducing paper results

Each reported condition maps to a training command in §4 plus an evaluation
command in §5:

- **Baseline agent** — train with §4b (`NASimNetDHRL`), evaluate with §5 on
  `corp_100hosts_dynamic_test.v2.yaml` for the generalization numbers.
- **Defense-regime ablations** — evaluate the same checkpoint against
  `scenarios/regimes/ids_off/` and `scenarios/regimes/defended/`.

Metrics are logged per-epoch (`captured_avg` on the train/test splits) to the
run directory; `model_best.pt` holds the best-scoring checkpoint used for the
reported evaluation figures.

---

## Provenance & License

This artifact builds on the open-source **NASimEmu** framework (MIT-licensed),
which itself derives from the Network Attack Simulator. Upstream copyright
notices are retained unchanged in [LICENSE](LICENSE) and in the respective
source trees, as the MIT license requires. All modifications introduced for this
submission are released under the same MIT license.

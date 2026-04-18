# PRIMA: Incremental PageRank via Processing-In-Memory Acceleration

PRIMA is a cycle-accurate simulator for an incremental PageRank accelerator
built on top of HBM2 with Processing-In-Memory (PIM) compute units. It extends
[PIMSimulator](https://github.com/SAITPublic/PIMSimulator) (itself built on
[DRAMSim2](https://github.com/umd-memsys/DRAMSim2)) with a PageRank workload,
incremental (warm-start) update logic, and head-to-head CPU-vs-PIM stats
collection.

**Authors:** Tobias Alam, Kidus Simegne, Viraj Shah, Arnav Devalapally, Jack Brady
*(EECS 573, University of Michigan)*

## Contents

1. [Overview](#1-overview)
2. [Hardware model](#2-hardware-model)
3. [Setup](#3-setup)
4. [Running PageRank on PIM](#4-running-pagerank-on-pim)
5. [Understanding the output](#5-understanding-the-output)
6. [Graph parameters](#6-graph-parameters)
7. [Acknowledgements](#7-acknowledgements)
8. [Citation](#8-citation)
9. [License](#9-license)

---

## 1. Overview

PageRank repeatedly computes `rank_new = (1 - d)/N + d * M * rank_old`. The
expensive step is the matrix-vector product `M * rank_old`. PRIMA offloads
that product to a PIM GEMV kernel running inside HBM2, and adds incremental
support so that an edge insertion re-uses the previous ranks (warm start)
instead of recomputing from scratch.

Two versions are implemented side-by-side:

|                       | CPU baseline                       | PIM accelerated                    |
|-----------------------|------------------------------------|------------------------------------|
| Precision             | FP32                               | FP16                               |
| SpMV step             | CPU (sparse, iterates edges)       | PIM GEMV kernel (dense N×N matrix) |
| Damping step          | CPU                                | CPU                                |
| Incremental updates   | Warm start from previous ranks     | Warm start from previous ranks     |

## 2. Hardware model

Inherited from PIMSimulator. Key bits:

- HBM2 stack, pseudo-channel mode, 1 PIM block per 2 banks.
- PIM ISA: `ADD`, `MUL`, `MAC`, `MAD`, `MOV`, `FILL`, `NOP`, `JUMP`, `EXIT`.
- Register files: CRF (command), GRF (vector), SRF (scalar).
- Address mapping: `Scheme8` (required for PIM mode).
- Config: [system_hbm.ini](system_hbm.ini), [ini/HBM2_samsung_2M_16B_x64.ini](ini/HBM2_samsung_2M_16B_x64.ini).

See the PIMSimulator README history for the full ISA / mode-transition
reference. PRIMA only drives the simulator through the existing PIM GEMV
kernel plus bare read/write transactions for the CPU-through-DRAM comparison.

## 3. Setup

### 3.1 Prerequisites

```bash
sudo apt install scons libgtest-dev
```

### 3.2 Build

```bash
scons
```

Build flags:

- `NO_STORAGE=1` — build without data-storage mode (faster, no correctness
  checks against stored data).
- `NO_EMUL=1` — skip the `tools/emulator_api` sources.

## 4. Running PageRank on PIM

```bash
# All PageRank tests
./sim --gtest_filter="PageRankFixture*"

# Specific test
./sim --gtest_filter="PageRankFixture.baseline_known_graph"
./sim --gtest_filter="PageRankFixture.stats_pim_vs_cpu"

# Only real-world graph tests (require dataset files in src/tests/)
./sim --gtest_filter="PageRankFixture.real_world*"
```

> PIM tests simulate cycle-accurate HBM2 memory, so they are slow on a laptop
> (synthetic tests take 5–15 s each; real-world tests with N=512 take 30–60 s).

### 4.1 Source layout

| File                                                                   | Purpose                                                                    |
|------------------------------------------------------------------------|----------------------------------------------------------------------------|
| [src/tests/PageRankGraph.h](src/tests/PageRankGraph.h)                 | Graph data structure, CPU PageRank, transition-matrix builder for PIM      |
| [src/tests/PageRankTestCases.cpp](src/tests/PageRankTestCases.cpp)     | gtest suite: correctness, incremental updates, stats comparison            |
| [src/tests/PIMKernel.cpp](src/tests/PIMKernel.cpp)                     | PIM GEMV / eltwise kernel driver (from PIMSimulator, unchanged)            |
| [src/tests/KernelAddrGen.cpp](src/tests/KernelAddrGen.cpp)             | Address generator for PIM transactions                                     |
| [src/tests/cit-HepPh.txt](src/tests/cit-HepPh.txt)                     | Real-world citation graph dataset (SNAP cit-HepPh)                         |
| [src/tests/web-Google.txt](src/tests/web-Google.txt)                   | Real-world web graph dataset (SNAP web-Google)                             |
| [src/tests/roadNet-CA.txt](src/tests/roadNet-CA.txt)                   | Real-world road network dataset (SNAP roadNet-CA)                          |

### 4.2 Test descriptions

| Test                                  | What it does                                                                                                                          |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `baseline_known_graph`                | CPU PageRank on a 4-node graph. Checks ranks sum to 1 and are positive.                                                               |
| `baseline_incremental_edges`          | 64-node ring graph, inserts 10 random edges, verifies warm-start converges to the same result as cold-start.                          |
| `baseline_random_graph_timing`        | CPU PageRank on a random 256-node graph (avg_deg=8), reports wall-clock time.                                                         |
| `pim_spmv_matches_cpu`                | One SpMV step on PIM, compared against CPU. Passes if all 256 outputs agree within 5% (FP16 tolerance).                               |
| `pim_full_pagerank`                   | Full PageRank to convergence using PIM for each SpMV step. Compares final ranks to CPU.                                               |
| `pim_incremental_edge_insertion`      | Inserts 16 edges mid-run into a 256-node graph (avg_deg=6); compares cold vs warm restart iteration counts on PIM.                    |
| `stats_pim_vs_cpu`                    | Side-by-side stats table (cycles, memory txns, data moved, bandwidth). CPU traffic estimated analytically.                            |
| `stats_cpu_dram_simulated`            | CPU dense-matrix access pattern routed through the same HBM2 DRAM simulator (no PIM ops). Apples-to-apples simulated cycle counts.    |
| `real_world_incremental_pagerank`     | cit-HepPh: cold-start on first 85% of edges, warm (delta) restart on remaining 15%. Compared to CPU baseline.                         |
| `real_world_web_google_pagerank`      | Same two-stage cold/warm benchmark on web-Google (capped at 512 nodes for simulation).                                                |
| `real_world_roadnet_pagerank`         | Same two-stage cold/warm benchmark on roadNet-CA (capped at 512 nodes for simulation).                                                |

## 5. Understanding the output

### 5.1 Correctness tests

```
[  PASSED  ] PageRankFixture.baseline_known_graph
```

Passing = ranks numerically correct within tolerance.

### 5.2 Incremental test output

```
Initial convergence:        12 iters
Cold restart (post-insert): 14 iters
Warm restart (incremental):  9 iters
```

- **Cold restart**: recompute from uniform ranks after edge insertion.
- **Warm restart**: recompute starting from the previous ranks (incremental).
- Fewer iterations for warm restart = the benefit of incremental PageRank.

### 5.3 Stats table — estimated CPU traffic (`stats_pim_vs_cpu`)

```
╔══════════════════════════════════════════════════════╗
║          Stats Comparison  (N=256)                   ║
╠══════════════════════╦═══════════════════════════════╣
║ Metric               ║ CPU baseline  │ PIM           ║
╠══════════════════════╬═══════════════════════════════╣
║ Iterations           ║             8 │             8 ║
║ FLOPs (M)            ║          0.03 │             - ║
║ Simulated cycles     ║             - │         44154 ║
║ Simulated time (ns)  ║             - │       44154.0 ║
║ Memory reads (txns)  ║             - │        104960 ║
║ Memory writes (txns) ║             - │        676480 ║
║ Data moved (MB)      ║          0.08 │         23.85 ║
║ Memory BW (GB/s)     ║             - │        566.34 ║
╚══════════════════════╩═══════════════════════════════╝
```

Dashes on the CPU side: the PIM simulator has no visibility into host DRAM.
`stats_cpu_dram_simulated` fixes that.

### 5.4 Stats table — both sides through DRAM simulator (`stats_cpu_dram_simulated`)

```
╔══════════════════════════════════════════════════════════════╗
║     Stats Comparison (N=256, both through HBM2 DRAM sim)    ║
╠══════════════════════════╦═══════════════╦═════════════════╣
║ Metric                   ║ CPU (no PIM)  ║ PIM             ║
╠══════════════════════════╬═══════════════╬═════════════════╣
║ Iterations               ║             8 ║              10 ║
║ Simulated cycles         ║          1392 ║           44154 ║
║ Simulated time (ns)      ║        1392.0 ║         44154.0 ║
║ Memory reads (txns)      ║          2536 ║          104960 ║
║ Memory writes (txns)     ║           256 ║          676480 ║
║ Data moved (MB)          ║          0.09 ║           23.85 ║
║ Memory BW (GB/s)         ║         64.18 ║          566.34 ║
╠══════════════════════════╩═══════════════╩═════════════════╣
║ PIM cycle speedup: 0.03x                                    ║
╚════════════════════════════════════════════════════════════╝
```

| Field                    | Meaning                                                                   |
|--------------------------|---------------------------------------------------------------------------|
| Simulated cycles         | Clock cycles in the simulated HBM2 hardware (not wall-clock time)         |
| Simulated time (ns)      | `cycles × tCK` — time this would take on real HBM2                        |
| Memory reads/writes      | Number of 32-byte burst transactions across all 64 channels               |
| Data moved (MB)          | `transactions × 32 bytes`                                                 |
| Memory BW (GB/s)         | Effective bandwidth utilized (HBM2 peak ≈ 900 GB/s)                       |
| PIM cycle speedup        | `CPU simulated time / PIM simulated time`                                 |

**Key takeaway.** For a sparse graph (N=256, ~2000 edges) the CPU baseline is
currently ~32× faster in simulated cycles because it only reads the edges it
needs (~2500 transactions) while PIM loads the entire dense 256×256 matrix
(~780k transactions). PIM does saturate ~566 GB/s (near HBM2 peak), showing the
hardware works as intended — the bottleneck is the dense matrix
representation. A sparse PIM kernel is the planned next optimization.

## 6. Graph parameters

Synthetic tests generate graphs internally — no external files needed.
Real-world tests require the dataset files to be present in `src/tests/`.
Graph parameters are hardcoded per test:

| Test                              | N (vertices)         | Avg out-degree             | RNG seed | Source                                                 |
|-----------------------------------|----------------------|----------------------------|----------|--------------------------------------------------------|
| `baseline_known_graph`            | 4                    | hand-crafted (5 edges)     | —        | synthetic                                              |
| `baseline_incremental_edges`      | 64                   | 1 (ring) + 10 random edges | 99       | synthetic                                              |
| `baseline_random_graph_timing`    | 256                  | 8                          | 42       | synthetic                                              |
| `pim_spmv_matches_cpu`            | 256                  | 8                          | 7        | synthetic                                              |
| `pim_full_pagerank`               | 256                  | 8                          | 13       | synthetic                                              |
| `pim_incremental_edge_insertion`  | 256                  | 6 (+ 16 inserted)          | 17 / 55  | synthetic                                              |
| `stats_pim_vs_cpu`                | 256                  | 8                          | 42       | synthetic                                              |
| `stats_cpu_dram_simulated`        | 256                  | 8                          | 42       | synthetic                                              |
| `real_world_incremental_pagerank` | ≤512 (padded to 16×) | real (85%/15% split)       | —        | [src/tests/cit-HepPh.txt](src/tests/cit-HepPh.txt)     |
| `real_world_web_google_pagerank`  | ≤512 (padded to 16×) | real (85%/15% split)       | —        | [src/tests/web-Google.txt](src/tests/web-Google.txt)   |
| `real_world_roadnet_pagerank`     | ≤512 (padded to 16×) | real (85%/15% split)       | —        | [src/tests/roadNet-CA.txt](src/tests/roadNet-CA.txt)   |

To change graph size or density for synthetic tests, edit the `const int N`
and `avg_deg` values at the top of each test in
[src/tests/PageRankTestCases.cpp](src/tests/PageRankTestCases.cpp), then
rebuild with `scons`. For real-world tests, change `MAX_NODES` (must be a
multiple of 16; `N % 16` padding applies automatically otherwise).

Benchmark plots from our runs live in [src/tests/figures/](src/tests/figures/);
the aggregation script is [src/tests/plot_benchmarks.py](src/tests/plot_benchmarks.py)
and raw numbers in [src/tests/benchmark_results.csv](src/tests/benchmark_results.csv).

## 7. Acknowledgements

PRIMA is built on top of two prior open-source simulators:

- **PIMSimulator** — Samsung Electronics Co. LTD. Provides the HBM2 + PIM
  block model, PIM ISA, and the GEMV / eltwise kernel drivers we re-use for
  the SpMV step. See [LICENSE-PIMSimulator](LICENSE-PIMSimulator).
- **DRAMSim2** — Elliott Cooper-Balis, Paul Rosenfeld, Bruce Jacob
  (University of Maryland). Provides the cycle-accurate DRAM model that
  PIMSimulator (and therefore PRIMA) is based on.
  See [LICENSE-DRAMSIM2](LICENSE-DRAMSIM2).

Real-world graph datasets are from the
[Stanford SNAP collection](https://snap.stanford.edu/data/): `cit-HepPh`,
`web-Google`, `roadNet-CA`.

Developed as the final project for **EECS 573 — Microarchitecture**,
University of Michigan.

## 8. Citation

If you use PRIMA in academic work, please cite:

```bibtex
@misc{prima2026,
  title        = {PRIMA: Incremental PageRank via Processing-In-Memory Acceleration},
  author       = {Alam, Tobias and Simegne, Kidus and Shah, Viraj and
                  Devalapally, Arnav and Brady, Jack},
  year         = {2026},
  howpublished = {EECS 573 Final Project, University of Michigan},
  note         = {Built on PIMSimulator (Samsung) and DRAMSim2 (UMD)}
}
```

Please also cite the underlying simulators:

```bibtex
@inproceedings{dramsim2,
  title     = {DRAMSim2: A Cycle Accurate Memory System Simulator},
  author    = {Rosenfeld, Paul and Cooper-Balis, Elliott and Jacob, Bruce},
  booktitle = {IEEE Computer Architecture Letters},
  year      = {2011}
}

@inproceedings{pimsimulator,
  title     = {HBM-PIM: A Function-in-Memory (FIM) Architecture for AI/ML Acceleration},
  author    = {Kang, Shin-haeng and others},
  booktitle = {Samsung Electronics},
  year      = {2021},
  note      = {PIMSimulator, \url{https://github.com/SAITPublic/PIMSimulator}}
}
```

## 9. License

PRIMA's own contributions are released under the MIT License — see
[LICENSE](LICENSE). The upstream components retain their original licenses:

- [LICENSE-DRAMSIM2](LICENSE-DRAMSIM2) — BSD 3-clause (UMD).
- [LICENSE-PIMSimulator](LICENSE-PIMSimulator) — Samsung, non-commercial
  academic/research use only.

Use of the combined work is subject to the most restrictive applicable
license (currently the PIMSimulator terms: non-commercial academic use only).

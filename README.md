# PrefGNN

Experimental code for large-scale graph neural network (GNN) training and feature caching. The project studies future-batch visibility windows, hot-feature caching, hot/cold node partitioning, and the effects of sampling and feature transfer on training performance. Supported datasets include IGB, Reddit, OGBN-Papers100M, and MAG240M; models include GraphSAGE, GCN, and GAT.

## Code Overview

| Component | Files / Directories | Purpose |
| --- | --- | --- |
| PrefGNN training | `ay_project/igb_gpu.py`, `mag_gpu.py`, `paper_gpu.py`, `redd.py` | Dataset-specific training with feature caches and batch queues; profiles sampling, transfer, and computation |
| Hot/cold partitioning | `ay_project/*_gate.py` | Partitions nodes by access hotness and applies corresponding feature retrieval and caching workflows |
| Single-NUMA experiments | `ay_project/*_single_numa.py` | Controls the NUMA node and thread counts for the main and cache processes |
| Cache services | `ay_project/storage_cpu3.py`, `storage_cpu_gate.py`, `storage_cpu3_single_numa.py` | Feature caching, window queues, interprocess feature transfer, and background cache workers |
| Cache policy comparison | `ay_project/storage_cpu_all_{igb,reddit,paper,mag}.py` | Compares random, degree, and hotmap initialization with future-window PreFC policies; reports miss rates |
| Window sensitivity | `ay_project/cache_hit_vs_k.py` | Replays the same sampled epoch across window sizes K, without model training |
| Queue memory experiments | `ay_project/queue_memory_vs_k.py` | Measures CPU/GPU storage for sampled-block queues across K values and estimates CPU ring-buffer capacity |
| Full-GPU reference | `ay_project/reddit_gpu_only.py` | Runs Reddit sampling, feature access, and training on the GPU, with individual or sequential model runs |
| Models and data loading | `ay_project/models/`, `igb_dataloader.py` in both directories | GNN implementations and IGB graph loading |
| Unified baselines | `dgl_Framework/benchmark_all_baselines.py` | Compares no-cache, random, degree, hotmap, BGL, and GIDS policies with end-to-end and per-stage timing |
| BGL / GIDS hit rates | `dgl_Framework/bgl_cache_hit.py`, `gids_cache_hit.py` | CPU cache-policy evaluation without full feature loading or model training |
| Baseline utilities | `dgl_Framework/bgl_core.py`, `bgl_data.py`, `bgl_structure_data.py` | BGL proximity-aware seed ordering, FIFO caching, and full-feature or structure-only data loading |
| Legacy DGL entries | `dgl_Framework/{igb,redd,paper,mag}.py` and `_gpu.py` variants | Dataset-specific training and profiling; requires the missing `storage.py` module |

Filename suffixes indicate GPU sampling (`_gpu`), hot/cold partitioning (`_gate`), and single-NUMA constraints (`_single_numa`). Sampling settings, cache ratios, and timing methods can differ between scripts; align these before comparing results.

## Environment and Data

Training experiments target Linux with NVIDIA GPUs. `ay_project/env.yml` records the original environment: Python 3.9, PyTorch 2.4.0, DGL 2.4.0 (CUDA 12.1 build), NumPy 1.23.5, OGB 1.3.6, and dependencies including CuPy.

```bash
conda env create -n prefgnn -f ay_project/env.yml
conda activate prefgnn
```

The environment file contains Linux-specific packages and CUDA builds. Configure package sources that provide these builds and ensure compatibility with your driver. It is not a portable Windows environment specification.

Datasets are not included. Prepare them before running:

| Dataset | Path configuration |
| --- | --- |
| IGB | Most entries accept `--path` for the root containing paths such as `large/processed/...` for edges, labels, and features. Some experiments require `large` (100 million nodes). |
| Reddit | Training entries generally use DGL `RedditDataset`. BGL/GIDS hit-rate entries accept `--reddit-root` pointing to a directory containing `reddit_graph.npz` and `reddit_data.npz`. |
| Papers100M | Newer entries accept `--paper-root` for the OGB data root; some legacy entries hardcode paths. |
| MAG240M | Newer entries accept `--mag-root`. Legacy entries contain paths such as `MAG240MDataset(root="/data")`; update these for your machine. |

Topology, sampled traces, and labels can consume substantial CPU memory even when features are not loaded. Replace `/your/data/...` below with actual paths. Commands use Bash syntax.

## Running Experiments

### 1. PrefGNN Training and Hot/Cold Partitioning

Resolve the extension and path requirements below, then run from `ay_project`:

```bash
cd ay_project

# IGB: GPU sampling with hot/cold partitioning
python igb_gpu_gate.py --mode puregpu --path /your/data/IGB-dataset --vis 6 --epochs 2

# MAG: GPU sampling with hot/cold partitioning (update the MAG path in the code first)
python mag_gpu_gate.py --mode puregpu --vis 5 --avg_input_size 250000

# Single-NUMA Reddit experiment
python redd_single_numa.py --mode puregpu --gpu 0 --numa-node 0 --main-threads 8 --cache-threads 12 --model gcn --cache-init hotmap

# Full-GPU Reddit reference (requires sufficient GPU memory)
python reddit_gpu_only.py --gpu 0 --model all
```

Key arguments: `--vis` sets the future-batch visibility window, `--epochs` sets the training epoch count, `--avg_input_size` configures feature-buffer capacity, and `--log_time True` enables per-stage timing. Single-NUMA entries also support `--model sage|gcn|gat` and `--cache-init hotmap|degree`.


### 2. Cache Policies and Window K

Run from `ay_project`:

```bash
# Original policy comparison: random, degree, hotmap, and PreFC
python storage_cpu_all_igb.py --path /your/data/IGB-dataset --dataset_size large

# Effect of K on hit rate (offline CPU replay, no model training)
python cache_hit_vs_k.py --dataset reddit
python cache_hit_vs_k.py --dataset igb --path /your/data/IGB-dataset

# Sampled-block queue storage: start with 100 batches to check the configuration
python queue_memory_vs_k.py --dataset reddit --gpu 0 --max-batches 100
```

`cache_hit_vs_k.py` compares K = 1, 2, 4, 6, 8, 10, 12, and 14. It reserves half the cache for fixed hot entries and uses the future window for dynamic admission and eviction. The full epoch's access trace is stored in CPU memory. Results go to `cache_hit_vs_k_results.txt`.

`queue_memory_vs_k.py` writes to `queue_memory_vs_k_results.txt`. Omit `--max-batches` to measure a full epoch. Results distinguish queue tensor storage, allocator diagnostics, and estimated buffer capacity; they do not represent total peak training GPU memory.

The original `storage_cpu_all_*.py` scripts generally append results to `missrate_compare_*.txt` in the working directory. Cache ratios and related settings are configured in the code.

### 3. DGL / BGL / GIDS Baselines

From the repository root, enter `dgl_Framework`:

```bash
cd dgl_Framework

# Reddit baselines: no cache, random, degree, hotmap, BGL, and GIDS
python benchmark_all_baselines.py --dataset reddit --gpu 0 --epochs 1

# Compare only BGL and GIDS
python benchmark_all_baselines.py --dataset reddit --methods bgl gids --gpu 0

# IGB training baselines
python benchmark_all_baselines.py --dataset igb --path /your/data/IGB-dataset --gpu 0

# Miss-rate evaluation only: no full feature loading or GPU training
python bgl_cache_hit.py --dataset reddit --reddit-root /your/data/reddit
python gids_cache_hit.py --dataset reddit --reddit-root /your/data/reddit --windows 8
```

The unified training entry requires no custom CUDA IPC extensions. Static cache policies run GraphSAGE; BGL/GIDS run GraphSAGE, GCN, and GAT sequentially. `--epochs 1` specifies one measurement pair: one end-to-end epoch and one per-stage timing epoch, for two training epochs in total. The default output is `baseline_breakdown_results.txt`.

# REPAIR

**Refining Exposure Pools via Adaptive Iterative Review**

Implementation of REPAIR and the nine baselines evaluated in the paper.

## Installation

The implementation uses Python 3.12, PyTorch 2.11.0 with CUDA 12.8, PyG 2.7.0, and Transformers 5.14.1.

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.11.0+cu128.html
pip install -r requirements.txt
```

Run commands from the repository root. GPU training uses the first visible CUDA device. Set `CUDA_VISIBLE_DEVICES` to select a physical GPU.

## Repository structure

```text
src/          REPAIR training, inference, evaluation, and data preprocessing
baselines/    Text kNN, GNNSafe, LLM-OOD OE, GOE-Identifier, GRASP,
              GOLD, CFC-A, LECT, and LG-Plug
configs/      Dataset, training, and baseline configurations
prompts/      Binary ID/OOD annotation prompts
```

## Datasets and preprocessing

Dataset identifiers, in paper order, are `wikics`, `cora_full`, `books_history`, `ele_photo`, `ele_computers`, `arxiv`, and `sportsfit`.

Place preprocessed datasets in `data/<identifier>/`. Data, model weights, and annotation caches are stored separately from the source code. Each dataset follows this format.

```text
graph.npz                 node_ids [N], x [N,384], edge_index [2,E]
seed0.npz                 targets [N], L, V, A, B, Q
seed1.npz
seed2.npz
id_definitions.json       [{id: "0", name: ..., definition: ...}, ...]
node_texts.json            node_id -> original node text
annotations/seed0.json    Fixed candidate annotations, likewise seed1/seed2
manifest.json             Relative public file paths and SHA256 hashes
sealed/truth.npz           node_ids, is_ood, class_targets
sealed/manifest.json       Evaluation file hashes
```

`x` contains float32 Nomic-384 embeddings. Edges are symmetric, unique, sorted, and exclude self-loops. `targets` contains contiguous ID class indices on `L` and `V`, with `-100` elsewhere. `is_ood=1` denotes OOD. `class_targets` uses the same ID class indices and `-100` for OOD. Training reads public inputs, and evaluation reads sealed truth after predictions have been saved.

### Preparing raw data

Prepare `raw.npz` with `node_ids`, `edge_index`, and integer `raw_labels`. Supply `texts.json` mapping node identifiers to original texts, and `classes.json` mapping original class indices to objects with `name` and `definition` fields. Then encode and preprocess the graph.

```bash
python -m src.encode --graph raw.npz --texts texts.json --model models/nomic --output features.npz
python -m src.preprocess --raw raw.npz --features features.npz --texts texts.json --classes classes.json --output data/new_graph
```

Use a local Nomic-Embed-Text-v1.5 model with revision `e9b6763023c676ca8431644204f50c2b100d9aab`. Encoding uses the `classification: ` prefix, an 8192-token limit, attention-mask mean pooling, LayerNorm over 768 coordinates, the first 384 coordinates, and L2 normalization. Inputs longer than 2048 tokens are encoded individually. Truncation counts are recorded for original texts exceeding 8192 tokens.

Preprocessing removes classes with fewer than 160 nodes, retains the induced graph, and canonicalizes edges. Each ID class supplies 100 training nodes and 50 validation nodes. A mixed pool of 2000 candidates is sampled and divided into disjoint folds A/B. Remaining nodes form Q. Generic preprocessing uses class seed 11, node seeds 100/101/102, fold seed 23, and half the retained classes as ID. Reproducing the paper requires the original fixed splits, including six ID classes for WikiCS. Add a backbone entry to `configs/datasets.json` for a new dataset.

### Importing fixed annotations

REPAIR, LLM-OOD OE, GOE-Identifier, and CFC-A share fixed candidate annotations. Prompts are in `prompts/`. Each seed must contain exactly one record per node in A∪B.

```json
{"node_id":"n123","status":"valid","annotation":{"decision":"OOD"}}
```

ID decisions may be `IND` or `ID`. Failed responses retain their identifiers and a non-valid status, with a null annotation if needed. They are excluded from exposure selection. Import files named `seed0.json`, `seed1.json`, and `seed2.json`.

```bash
python -m src.import_annotations --data data/new_graph --annotations annotation_results
```

## REPAIR training and evaluation

```bash
python -m src.train --dataset wikics --data data/wikics --seed 0 --output runs/repair/wikics/seed0
python -m src.test --data data/wikics --seed 0 --run runs/repair/wikics/seed0
```

Training first fits an ID-only M0. An existing initialization for the same dataset, backbone, and seed can be supplied with `--m0 path/to/M0.pt`. REPAIR performs eight rounds. Each teacher trains on exposures from one fold and reviews the opposite fold using raw GNN Energy. Both teachers update before new pools are committed. Teacher model, optimizer, and RNG states continue across rounds. Review permits withdrawal and recovery while keeping LLM annotations fixed. Only labeled ID nodes supply classification targets.

The final detector starts from M0 with a fresh optimizer and the merged exposure pool. Its complete Energy-plus-text head score is propagated with K=3 and alpha=0.5. Parameters are in `configs/repair.json` and `configs/datasets.json`. The backbone is GCN-128, with first-layer BN on CoraFull and GCN-256 with first-layer BN on ogbn-arxiv. M0 uses minimum ID-validation CE selection. Exposure stages use the minimum regularized joint objective, including epoch 0, with at most 2000 epochs and patience 20.

Recompute predictions from the final checkpoint and fitted head.

```bash
python -m src.predict --dataset wikics --data data/wikics --seed 0 --run runs/repair/wikics/seed0 --output runs/repair/wikics/seed0/recomputed.npz
```

`src.test` evaluates saved training predictions and reports AUROC, ID-positive AUPR, FPR95 at 95% ID recall, ID accuracy, and OOD-positive exposure F1. Per-seed metrics use fractions. After running seeds 0, 1, and 2, aggregate percentages and sample standard deviations.

```bash
python -m src.summarize --runs runs/repair/wikics
```

Missing seeds remain unavailable. Completed stages can be reused under the same configuration. Interrupted stages without continuation checkpoints are rerun.

## Baselines

```bash
python -m baselines.train --method gnnsafe --dataset wikics --data data/wikics --seed 0 --output runs/gnnsafe/wikics/seed0
python -m src.test --data data/wikics --seed 0 --run runs/gnnsafe/wikics/seed0
```

| Method argument | Implementation |
|---|---|
| `text_knn` | Tenth-nearest cosine distance to L, with float64 normalization. |
| `gnnsafe` | ID-only matched GCN, minimum validation CE, Energy propagation with K=3. |
| `llm_ood_oe` | Starts from M0, uses all valid LLM-OOD candidates, separate ID/OOD margin means, lambda=0.01, and joint-objective selection. |
| `goe_identifier` | Random initialization and original equal-length truncation loss. Uses the last model after 200 epochs, lambda=0.01 except Sportsfit with lambda=0.05. |
| `grasp` | ID-only GCN, L∪V anchors, MSP-based graph augmentation, K=3, alpha=0.5. |
| `gold` | Author GCN-64 with BN, diffusion generation, and joint training. Uses arxiv/label on ogbn-arxiv and cora/label elsewhere, with the author's final-model selection and internal propagation. |
| `cfc` | CFC-A with fixed annotations, original denoising and boundary mixup, 512/128 hidden dimensions without BN or activation, SGD, minimum validation CE, and propagated OOD-class probability. |
| `lect` | Synthetic nodes, 128-dimensional projection, GCN-64 with BN, original pair/triplet losses, original-graph validation/inference, and logit centering only for scoring. |
| `lgplug` | Graph-text alignment, candidate clustering, dynamic-codebook queries, and a randomly initialized matched GCN with propagated Energy OE. |

GNNSafe, GRASP, and LLM-OOD OE accept `--m0` to share ID pretraining. GOE-Identifier and LG-Plug use random initialization. CFC-A, LECT, and GOLD retain their method-specific architectures. CFC-A, LECT, and LG-Plug use at least 1000 epochs, patience 300, min_delta=0.0001, and a 5000-epoch cap. LG-Plug uses lambda=0.01. GOLD output directories must end in `seed0`, `seed1`, or `seed2`.

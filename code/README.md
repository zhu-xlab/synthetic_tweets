# Geographic MIL

Runnable Multiple Instance Learning and Single Instance Learning baselines for
the `synthetic_tweets` dataset.

This folder is meant to be placed under the repository `code/` directory. The
training scripts automatically look for `../data/synthetic_tweets.csv` when run
from `code/`.

## Models

- `MIL_Max.py`: MIL with max pooling over instance logits.
- `MIL_Mean.py`: MIL with mean pooling over instance representations.
- `MIL_Attention.py`: attention-based MIL.
- `MIL_DeepSets.py`: DeepSets-style MIL.
- `SIL_MajorityVote.py`: single-instance classifier with building-level majority vote.
- `SIL_ProbabilityAverage.py`: single-instance classifier with building-level probability averaging.

## Quick Start

From the `synthetic_tweets` repository root:

```bash
python code/run_all_models.py --epochs 1 --max-buildings 40 --device cpu
```

Or from inside the `code/` directory:

```bash
python run_all_models.py --epochs 1 --max-buildings 40 --device cpu
```

Run one model:

```bash
python code/MIL_Attention.py --epochs 2 --max-buildings 80 --device cpu
```

Outputs are written under `outputs/<MODEL_NAME>/metrics.json`.

## Encoder Options

Default `--encoder mock` uses a small trainable embedding encoder with stable
hash tokenization. It is intended for smoke tests and reproducibility checks.

To use a Hugging Face transformer encoder:

```bash
python code/MIL_Max.py \
  --encoder bert \
  --pretrained-model google-bert/bert-base-multilingual-cased \
  --epochs 1 \
  --max-buildings 40
```

This requires the model to be available locally or downloadable.

## Data

The scripts expect a CSV with:

- `building_id`
- `label`, with values `residential` or `commercial`
- `tweet`

Use `data/synthetic_tweets.csv`. 

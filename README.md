# 🏙️ Synthetic Oracle Tweets Dataset for Building Function Classification

This dataset contains **synthetic, LLM-generated tweets** designed as an **oracle benchmark** for studying the effects of noise in weakly supervised, text-based building function classification (BFC) tasks. It includes **15,222 multilingual tweets** generated for **6,000 real-world buildings** across **41 global cities**.

---

## 📦CSV Dataset Contents

Each entry in the CSV file dataset corresponds to a building and includes:
- **Building Name**
- **City**
- **Functional Tag** (e.g., `restaurant`, `apartment`)
- **Tweet Language Distribution** (e.g., `["English", "English"]`)
- **Generated Tweets** (semantically aligned with the building’s function)

### 📝 The metadata of all buildings are saved also in the metadata.jsonl file, Example entry is a building:

```json
{
  "building_name": "Merlex Auto Group",
  "building_city": "WashingtonDC",
  "building_tag": "Retail",
  "tweets_distribution": ["English", "English"],
  "tweets": [
    "Bought new rims here at Merlex Auto yesterday, totally transformed my ride! #AutoCare",
    "Merlex Auto Group really knows how to treat car lovers right. The staff? Super knowledgeable."
  ]
}
```



```shell script
/synthetic_tweets/
│
├── data/       
│   ├── metadata.jsonl                  # Metadata used for tweet generation
│   └── synthetic_tweets.csv            # Public synthetic tweet dataset
│
├── code/
│   ├── tweets_generation.ipynb         # Code for tweets generation
│   ├── run_all_models.py               # Run all MIL/SIL baselines
│   ├── MIL_Attention.py
│   ├── MIL_DeepSets.py
│   ├── MIL_Max.py
│   ├── MIL_Mean.py
│   ├── SIL_MajorityVote.py
│   └── SIL_ProbabilityAverage.py
│
└── README.md   
```

## 🚀 Runnable MIL/SIL Baselines

This repository also includes training code for six building function
classification baselines under `code/`:

- MIL-Max
- MIL-Mean
- MIL-Attention
- MIL-DeepSets
- SIL with majority vote
- SIL with probability averaging

Run all models on the bundled synthetic dataset with a lightweight mock encoder:

```bash
python code/run_all_models.py --epochs 1 --max-buildings 40 --device cpu
```

Run a single model:

```bash
python code/MIL_Attention.py --epochs 2 --max-buildings 80 --device cpu
```

The default mock encoder is intended for smoke tests and reproducibility checks.
To use a Hugging Face encoder:

```bash
python code/MIL_Max.py \
  --encoder bert \
  --pretrained-model google-bert/bert-base-multilingual-cased \
  --epochs 1
```

Metrics are written to `outputs/<MODEL_NAME>/metrics.json`.

Note: the runnable baselines use `data/synthetic_tweets.csv` by default. Files
containing real tweets are intentionally excluded from this public release.





## 🔗 Resources
 - 📜 Project Paper (Preprint) [Project Paper (Preprint)](http://arxiv.org/abs/2503.22856)
 - 🤗 LLM Used: LLaMA-3.3-70B-Instruct (bnb-4bit) [LLM Used: LLaMA-3.3-70B-Instruct (bnb-4bit)](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-bnb-4bit)



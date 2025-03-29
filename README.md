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


```shell script
echo "hello"
```

## 🔗 Resources
 - 📜 Project Paper (Preprint) [Project Paper (Preprint)]()
 - 🤗 LLM Used: LLaMA-3.3-70B-Instruct (bnb-4bit) [LLM Used: LLaMA-3.3-70B-Instruct (bnb-4bit)](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-bnb-4bit)


## 📄 Citation

If you use this dataset in your work, please cite it as:

```bibtex
@misc{syntheticBFC2025,
  author       = {Shanshan Bai, Anna Kruspe, Xiaoxiang Zhu},
  title        = {Generating Synthetic Oracle Datasets to Analyze Noise Impact: A Study on Building Function Classification Using Tweets},
  year         = {2025}
}
```

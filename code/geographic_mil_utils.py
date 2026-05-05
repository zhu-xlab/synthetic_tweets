#!/usr/bin/env python3
"""Shared utilities for runnable Geographic MIL/SIL examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


LABEL_TO_ID = {"residential": 0, "commercial": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
MODULE_DIR = Path(__file__).resolve().parent


@dataclass
class RunConfig:
    data_path: Path
    output_dir: Path
    encoder: str
    pretrained_model: str
    cache_dir: str | None
    max_len: int
    batch_size: int
    epochs: int
    lr: float
    seed: int
    device: torch.device
    hidden_size: int
    vocab_size: int
    max_buildings: int | None
    freeze_encoder: bool


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-path", type=Path, default=default_data_path())
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--encoder", choices=["mock", "bert"], default="mock")
    parser.add_argument("--pretrained-model", default="google-bert/bert-base-multilingual-cased")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--max-buildings", type=int, default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    return parser


def default_data_path() -> Path:
    candidates = [
        Path("synthetic_tweets.csv"),
        MODULE_DIR / "synthetic_tweets.csv",
        MODULE_DIR.parent / "data" / "synthetic_tweets.csv",
        MODULE_DIR.parent.parent / "data" / "synthetic_tweets.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def parse_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        encoder=args.encoder,
        pretrained_model=args.pretrained_model,
        cache_dir=args.cache_dir,
        max_len=args.max_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=torch.device(args.device),
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        max_buildings=args.max_buildings,
        freeze_encoder=args.freeze_encoder,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_synthetic_data(path: Path, max_buildings: int | None = None) -> pd.DataFrame:
    if not path.exists():
        fallback = default_data_path()
        if fallback.exists():
            path = fallback
    df = pd.read_csv(path)
    required = {"building_id", "label", "tweet"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    df = df[df["label"].isin(LABEL_TO_ID)].copy()
    df["tweet"] = df["tweet"].fillna("").astype(str)
    df["label_id"] = df["label"].map(LABEL_TO_ID).astype(int)
    if max_buildings:
        keep = df["building_id"].drop_duplicates().head(max_buildings)
        df = df[df["building_id"].isin(keep)].copy()
    return df.reset_index(drop=True)


def split_by_building(df: pd.DataFrame, seed: int):
    building_labels = df.groupby("building_id")["label_id"].first().reset_index()
    stratify = stratify_if_possible(building_labels["label_id"])
    train_ids, tmp_ids = train_test_split(
        building_labels,
        test_size=0.3,
        random_state=seed,
        stratify=stratify,
    )
    tmp_stratify = stratify_if_possible(tmp_ids["label_id"])
    val_ids, test_ids = train_test_split(
        tmp_ids,
        test_size=0.5,
        random_state=seed,
        stratify=tmp_stratify,
    )
    return (
        df[df["building_id"].isin(train_ids["building_id"])].copy(),
        df[df["building_id"].isin(val_ids["building_id"])].copy(),
        df[df["building_id"].isin(test_ids["building_id"])].copy(),
    )


def stratify_if_possible(labels: pd.Series):
    counts = labels.value_counts()
    if len(counts) > 1 and counts.min() >= 2:
        return labels
    return None


def hash_tokenize(text: str, max_len: int, vocab_size: int):
    words = text.lower().split()[:max_len]
    ids = [stable_hash(w) % (vocab_size - 2) + 2 for w in words]
    mask = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(0)
        mask.append(0)
    return ids, mask


def stable_hash(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest(), 16)


class TextEncoder(nn.Module):
    def __init__(self, config: RunConfig):
        super().__init__()
        self.mode = config.encoder
        if config.encoder == "bert":
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(config.pretrained_model, cache_dir=config.cache_dir)
            self.hidden = self.model.config.hidden_size
        else:
            self.model = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
            self.hidden = config.hidden_size

    def forward(self, input_ids, attention_mask):
        if self.mode == "bert":
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            hidden = out.last_hidden_state
        else:
            hidden = self.model(input_ids)
        mask = attention_mask.unsqueeze(-1).type_as(hidden)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class TokenizerAdapter:
    def __init__(self, config: RunConfig):
        self.config = config
        self.tokenizer = None
        if config.encoder == "bert":
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model, cache_dir=config.cache_dir)

    def encode(self, texts: list[str]):
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_len,
                return_tensors="pt",
            )
            return encoded["input_ids"], encoded["attention_mask"]
        ids, masks = zip(*(hash_tokenize(t, self.config.max_len, self.config.vocab_size) for t in texts))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.long)


def make_mil_bags(df: pd.DataFrame):
    bags = defaultdict(list)
    labels = {}
    for row in df.itertuples(index=False):
        bags[row.building_id].append(row.tweet)
        labels[row.building_id] = int(row.label_id)
    ordered_ids = sorted(bags)
    return [bags[bid] for bid in ordered_ids], [labels[bid] for bid in ordered_ids], ordered_ids


class MILDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: TokenizerAdapter):
        bags, labels, building_ids = make_mil_bags(df)
        self.labels = labels
        self.building_ids = building_ids
        self.encoded = [tokenizer.encode(bag) for bag in bags]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_ids, attention_mask = self.encoded[idx]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "building_id": self.building_ids[idx],
        }


def pad_mil_batch(batch):
    max_k = max(item["input_ids"].shape[0] for item in batch)
    max_l = batch[0]["input_ids"].shape[1]
    ids, masks, labels, instance_masks = [], [], [], []
    for item in batch:
        k = item["input_ids"].shape[0]
        pad_k = max_k - k
        ids.append(torch.cat([item["input_ids"], torch.zeros((pad_k, max_l), dtype=torch.long)], dim=0))
        masks.append(torch.cat([item["attention_mask"], torch.zeros((pad_k, max_l), dtype=torch.long)], dim=0))
        labels.append(item["label"])
        instance_masks.append(torch.cat([torch.ones(k), torch.zeros(pad_k)]))
    return {
        "input_ids": torch.stack(ids),
        "attention_mask": torch.stack(masks),
        "instance_mask": torch.stack(instance_masks),
        "label": torch.stack(labels),
    }


class SILInstanceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: TokenizerAdapter):
        self.df = df.reset_index(drop=True)
        self.input_ids, self.attention_mask = tokenizer.encode(self.df["tweet"].tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": torch.tensor(int(row.label_id), dtype=torch.long),
            "building_id": row.building_id,
        }


def create_mil_loaders(config: RunConfig):
    df = load_synthetic_data(config.data_path, config.max_buildings)
    train_df, val_df, test_df = split_by_building(df, config.seed)
    tokenizer = TokenizerAdapter(config)
    return (
        DataLoader(MILDataset(train_df, tokenizer), batch_size=config.batch_size, shuffle=True, collate_fn=pad_mil_batch),
        DataLoader(MILDataset(val_df, tokenizer), batch_size=config.batch_size, shuffle=False, collate_fn=pad_mil_batch),
        DataLoader(MILDataset(test_df, tokenizer), batch_size=config.batch_size, shuffle=False, collate_fn=pad_mil_batch),
    )


def create_sil_loaders(config: RunConfig):
    df = load_synthetic_data(config.data_path, config.max_buildings)
    train_df, val_df, test_df = split_by_building(df, config.seed)
    tokenizer = TokenizerAdapter(config)
    return (
        DataLoader(SILInstanceDataset(train_df, tokenizer), batch_size=config.batch_size, shuffle=True),
        DataLoader(SILInstanceDataset(val_df, tokenizer), batch_size=config.batch_size, shuffle=False),
        DataLoader(SILInstanceDataset(test_df, tokenizer), batch_size=config.batch_size, shuffle=False),
    )


def train_mil_model(model, loaders, config: RunConfig, loss_fn=None):
    train_loader, val_loader, test_loader = loaders
    return train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        config,
        forward_fn=lambda m, b: m(
            b["input_ids"].to(config.device),
            b["attention_mask"].to(config.device),
            b["instance_mask"].to(config.device),
        ),
        loss_fn=loss_fn,
    )


def train_sil_model(model, loaders, config: RunConfig, aggregation: str):
    train_loader, val_loader, test_loader = loaders
    model.to(config.device)
    if config.freeze_encoder and hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = False
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.lr)
    best_state = None
    best_f1 = -1.0
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(config.device), batch["attention_mask"].to(config.device))
            loss = loss_fn(logits, batch["label"].to(config.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
        val = evaluate_sil(model, val_loader, config, aggregation)
        print(f"epoch={epoch} train_loss={total / max(1, len(train_loader)):.4f} val_f1={val['f1_macro']:.4f}")
        if val["f1_macro"] > best_f1:
            best_f1 = val["f1_macro"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    test = evaluate_sil(model, test_loader, config, aggregation)
    save_report(config, test)
    return test


def evaluate_sil(model, loader, config: RunConfig, aggregation: str):
    model.eval()
    grouped_logits = defaultdict(list)
    grouped_labels = {}
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(config.device), batch["attention_mask"].to(config.device))
            probs = torch.softmax(logits, dim=-1).cpu()
            preds = probs.argmax(dim=-1)
            building_ids = batch["building_id"].tolist() if hasattr(batch["building_id"], "tolist") else batch["building_id"]
            labels = batch["label"].cpu().tolist()
            for bid, label, pred, prob in zip(building_ids, labels, preds.tolist(), probs):
                grouped_labels[bid] = label
                grouped_logits[bid].append((pred, prob))
    y_true, y_pred = [], []
    for bid in sorted(grouped_labels):
        values = grouped_logits[bid]
        y_true.append(grouped_labels[bid])
        if aggregation == "majority":
            votes = defaultdict(int)
            for pred, _ in values:
                votes[pred] += 1
            y_pred.append(sorted(votes.items(), key=lambda item: (-item[1], item[0]))[0][0])
        elif aggregation == "probability_average":
            y_pred.append(torch.stack([prob for _, prob in values]).mean(dim=0).argmax().item())
        else:
            raise ValueError(f"Unknown SIL aggregation: {aggregation}")
    labels = [0, 1]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[ID_TO_LABEL[i] for i in labels],
        output_dict=True,
        zero_division=0,
    )
    f1 = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]
    return {"accuracy": accuracy_score(y_true, y_pred), "f1_macro": f1, "report": report}


def train_model(model, train_loader, val_loader, test_loader, config: RunConfig, forward_fn: Callable, loss_fn=None):
    model.to(config.device)
    if config.freeze_encoder and hasattr(model, "encoder"):
        for param in model.encoder.parameters():
            param.requires_grad = False
    loss_fn = loss_fn or nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.lr)
    best_state = None
    best_f1 = -1.0
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = forward_fn(model, batch)
            loss = loss_fn(logits, batch["label"].to(config.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
        val = evaluate_batches(model, val_loader, config, forward_fn)
        print(f"epoch={epoch} train_loss={total / max(1, len(train_loader)):.4f} val_f1={val['f1_macro']:.4f}")
        if val["f1_macro"] > best_f1:
            best_f1 = val["f1_macro"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    test = evaluate_batches(model, test_loader, config, forward_fn)
    save_report(config, test)
    return test


def evaluate_batches(model, loader, config: RunConfig, forward_fn: Callable):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            logits = forward_fn(model, batch)
            y_true.extend(batch["label"].cpu().tolist())
            y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
    labels = [0, 1]
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[ID_TO_LABEL[i] for i in labels],
        output_dict=True,
        zero_division=0,
    )
    f1 = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]
    return {"accuracy": accuracy_score(y_true, y_pred), "f1_macro": f1, "report": report}


def save_report(config: RunConfig, metrics: dict) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with open(config.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"test_accuracy={metrics['accuracy']:.4f} test_f1={metrics['f1_macro']:.4f}")
    print(f"saved metrics to {config.output_dir / 'metrics.json'}")

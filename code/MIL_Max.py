#!/usr/bin/env python3
"""MIL with max pooling over instance logits."""

import torch
import torch.nn as nn

from geographic_mil_utils import TextEncoder, build_arg_parser, create_mil_loaders, parse_config, set_seed, train_mil_model


class MILMax(nn.Module):
    def __init__(self, config, num_labels=2):
        super().__init__()
        self.encoder = TextEncoder(config)
        self.classifier = nn.Linear(self.encoder.hidden, num_labels)

    def forward(self, input_ids, attention_mask, instance_mask):
        batch_size, bag_size, seq_len = input_ids.shape
        flat_ids = input_ids.view(batch_size * bag_size, seq_len)
        flat_mask = attention_mask.view(batch_size * bag_size, seq_len)
        embeddings = self.encoder(flat_ids, flat_mask)
        logits = self.classifier(embeddings).view(batch_size, bag_size, -1)
        logits = logits.masked_fill(instance_mask.eq(0).unsqueeze(-1), torch.finfo(logits.dtype).min)
        return logits.max(dim=1).values


def main():
    parser = build_arg_parser("Train MIL-Max on synthetic geographic tweet data.")
    args = parser.parse_args()
    config = parse_config(args)
    config.output_dir = config.output_dir / "MIL_Max"
    set_seed(config.seed)
    metrics = train_mil_model(MILMax(config), create_mil_loaders(config), config)
    print(metrics["report"])


if __name__ == "__main__":
    main()

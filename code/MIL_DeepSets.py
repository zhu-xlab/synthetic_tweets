#!/usr/bin/env python3
"""DeepSets-style MIL for synthetic geographic tweet bags."""

import torch
import torch.nn as nn

from geographic_mil_utils import TextEncoder, build_arg_parser, create_mil_loaders, parse_config, set_seed, train_mil_model


class MILDeepSets(nn.Module):
    def __init__(self, config, num_labels=2, set_hidden=128):
        super().__init__()
        self.encoder = TextEncoder(config)
        hidden = self.encoder.hidden
        self.phi = nn.Sequential(nn.Linear(hidden, set_hidden), nn.ReLU(), nn.Dropout(0.1))
        self.rho = nn.Sequential(nn.Linear(set_hidden, set_hidden), nn.ReLU(), nn.Dropout(0.1), nn.Linear(set_hidden, num_labels))

    def forward(self, input_ids, attention_mask, instance_mask):
        batch_size, bag_size, seq_len = input_ids.shape
        flat_ids = input_ids.view(batch_size * bag_size, seq_len)
        flat_mask = attention_mask.view(batch_size * bag_size, seq_len)
        embeddings = self.encoder(flat_ids, flat_mask).view(batch_size, bag_size, -1)
        transformed = self.phi(embeddings) * instance_mask.unsqueeze(-1)
        return self.rho(transformed.sum(dim=1))


def main():
    parser = build_arg_parser("Train DeepSets MIL on synthetic geographic tweet data.")
    parser.add_argument("--set-hidden", type=int, default=128)
    args = parser.parse_args()
    config = parse_config(args)
    config.output_dir = config.output_dir / "MIL_DeepSets"
    set_seed(config.seed)
    metrics = train_mil_model(MILDeepSets(config, set_hidden=args.set_hidden), create_mil_loaders(config), config)
    print(metrics["report"])


if __name__ == "__main__":
    main()

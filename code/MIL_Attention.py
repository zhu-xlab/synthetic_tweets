#!/usr/bin/env python3
"""Attention-based MIL for synthetic geographic tweet bags."""

import torch
import torch.nn as nn

from geographic_mil_utils import TextEncoder, build_arg_parser, create_mil_loaders, parse_config, set_seed, train_mil_model


class MILAttention(nn.Module):
    def __init__(self, config, num_labels=2, attention_hidden=128):
        super().__init__()
        self.encoder = TextEncoder(config)
        hidden = self.encoder.hidden
        self.v = nn.Linear(hidden, attention_hidden)
        self.u = nn.Linear(hidden, attention_hidden)
        self.w = nn.Linear(attention_hidden, 1, bias=False)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, instance_mask):
        batch_size, bag_size, seq_len = input_ids.shape
        flat_ids = input_ids.view(batch_size * bag_size, seq_len)
        flat_mask = attention_mask.view(batch_size * bag_size, seq_len)
        embeddings = self.encoder(flat_ids, flat_mask).view(batch_size, bag_size, -1)
        attn_logits = self.w(torch.tanh(self.v(embeddings)) * torch.sigmoid(self.u(embeddings))).squeeze(-1)
        attn_logits = attn_logits.masked_fill(instance_mask.eq(0), torch.finfo(attn_logits.dtype).min)
        weights = torch.softmax(attn_logits, dim=1)
        bag_embedding = torch.sum(weights.unsqueeze(-1) * embeddings, dim=1)
        return self.classifier(bag_embedding)


def main():
    parser = build_arg_parser("Train attention MIL on synthetic geographic tweet data.")
    parser.add_argument("--attention-hidden", type=int, default=128)
    args = parser.parse_args()
    config = parse_config(args)
    config.output_dir = config.output_dir / "MIL_Attention"
    set_seed(config.seed)
    metrics = train_mil_model(MILAttention(config, attention_hidden=args.attention_hidden), create_mil_loaders(config), config)
    print(metrics["report"])


if __name__ == "__main__":
    main()

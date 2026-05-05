#!/usr/bin/env python3
"""Single-instance classifier with building-level probability averaging."""

import torch.nn as nn

from geographic_mil_utils import TextEncoder, build_arg_parser, create_sil_loaders, parse_config, set_seed, train_sil_model


class SILTextClassifier(nn.Module):
    def __init__(self, config, num_labels=2):
        super().__init__()
        self.encoder = TextEncoder(config)
        self.classifier = nn.Linear(self.encoder.hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        return self.classifier(self.encoder(input_ids, attention_mask))


def main():
    parser = build_arg_parser("Train SIL with building-level probability averaging.")
    args = parser.parse_args()
    config = parse_config(args)
    config.output_dir = config.output_dir / "SIL_ProbabilityAverage"
    set_seed(config.seed)
    metrics = train_sil_model(
        SILTextClassifier(config),
        create_sil_loaders(config),
        config,
        aggregation="probability_average",
    )
    print(metrics["report"])


if __name__ == "__main__":
    main()

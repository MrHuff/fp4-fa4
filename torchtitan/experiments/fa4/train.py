# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
"""Launch TorchTitan with the opt-in FA4 training integration."""

from torchtitan.train import main

from .trainer import FA4Trainer


if __name__ == "__main__":
    main(FA4Trainer)

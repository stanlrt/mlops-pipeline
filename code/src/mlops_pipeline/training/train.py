"""ResNet-18 training entrypoint. Stub — to be filled in."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    raise NotImplementedError(f"training loop not yet implemented (config={args.config})")


if __name__ == "__main__":
    main()

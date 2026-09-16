"""Explicit model preparation: python -m lama_inpaint download [--offline]."""

import argparse
import sys

from .model_store import ModelError, ensure_model


def main():
    """Verify/download the fixed artifact, returning a nonzero status on failure."""
    parser = argparse.ArgumentParser(description="Prepare verified LaMA model weights")
    parser.add_argument("command", choices=["download"])
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Verify cached weights without network access",
    )
    args = parser.parse_args()
    try:
        path = ensure_model(args.cache_dir, download=not args.offline)
    except (ModelError, OSError) as exc:
        print(f"LaMA model preparation failed: {exc}", file=sys.stderr)
        return 1
    print(f"LaMA model verified: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

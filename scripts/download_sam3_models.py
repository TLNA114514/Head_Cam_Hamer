#!/usr/bin/env python3
"""Download or verify the SAM3 checkpoints used by the head-camera pipeline."""

from __future__ import annotations

import argparse
import json

from huggingface_hub import hf_hub_download


MODELS = {
    "sam3": ("facebook/sam3", "sam3.pt"),
    "sam3.1": ("facebook/sam3.1", "sam3.1_multiplex.pt"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-only", action="store_true", help="Only verify the local Hugging Face cache.")
    parser.add_argument("--version", action="append", choices=sorted(MODELS), help="Defaults to both versions.")
    args = parser.parse_args()

    resolved = {}
    for version in args.version or list(MODELS):
        repo_id, filename = MODELS[version]
        config = hf_hub_download(repo_id=repo_id, filename="config.json", local_files_only=args.local_only)
        checkpoint = hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=args.local_only)
        resolved[version] = {"config": config, "checkpoint": checkpoint}
    print(json.dumps(resolved, indent=2))


if __name__ == "__main__":
    main()

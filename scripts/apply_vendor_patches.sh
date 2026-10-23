#!/usr/bin/env bash
# Apply the environment patches to vendored submodules (identical for every experimental condition).
#  - contact_graspnet: run inference under torch.no_grad() and free CUDA memory (avoids OOM on a 16 GB GPU).
#  - sam3: load checkpoints/sam3/sam3.pt (or $SAM3_CHECKPOINT_PATH) instead of the gated HF download.
set -euo pipefail
cd "$(dirname "$0")/.."
for p in patches/*.patch; do
  if patch -p1 -R --dry-run -s -f < "$p" >/dev/null 2>&1; then echo "already applied: $p"; else patch -p1 -s < "$p"; echo "applied $p"; fi
done

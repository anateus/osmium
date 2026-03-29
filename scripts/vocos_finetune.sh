#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR"

TRAINING_DIR="training"
DATA_DIR="$TRAINING_DIR/data"
CHECKPOINT_DIR="$TRAINING_DIR/checkpoints"
LOG_DIR="$TRAINING_DIR/logs"
MODEL_DIR="$TRAINING_DIR/models/vocos-mel-24khz-finetuned"

RESUME=""
if [[ "${1:-}" == "--resume" ]]; then
    RESUME="--resume"
    echo "==> Resume mode: will continue from last checkpoint"
fi

echo "============================================"
echo "  Vocos Fine-Tuning Pipeline"
echo "  $(date)"
echo "============================================"

echo ""
echo "==> Stage 1: Download LibriTTS train-clean-100"
.venv/bin/python scripts/vocos_finetune/download_data.py --data-dir "$DATA_DIR"

echo ""
echo "==> Stage 2: Verify training dependencies"
.venv/bin/python -c "import pytorch_lightning; import transformers; import einops" 2>/dev/null || {
    echo "Installing training dependencies..."
    uv pip install "pytorch-lightning==1.8.6" "transformers>=4.30" "einops"
}

echo ""
echo "==> Stage 3: Training (10k effective steps, ~3-5 hours)"
echo "    Checkpoints: $CHECKPOINT_DIR"
echo "    Logs: $LOG_DIR (tensorboard --logdir $LOG_DIR)"
echo "    Ctrl-C to pause, re-run with --resume to continue"
echo ""
.venv/bin/python -m scripts.vocos_finetune.train \
    --train-filelist "$DATA_DIR/filelists/train.txt" \
    --val-filelist "$DATA_DIR/filelists/val.txt" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log-dir "$LOG_DIR" \
    --max-steps 20000 \
    --batch-size 16 \
    --lr 2e-5 \
    $RESUME

echo ""
echo "==> Stage 4: Convert best checkpoint to MLX"
BEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/best-*.ckpt 2>/dev/null | head -1)
if [[ -z "$BEST_CKPT" ]]; then
    echo "No best checkpoint found, using last.ckpt"
    BEST_CKPT="$CHECKPOINT_DIR/last.ckpt"
fi
echo "    Using checkpoint: $BEST_CKPT"
.venv/bin/python -m scripts.vocos_finetune.convert_mlx \
    --checkpoint "$BEST_CKPT" \
    --output-dir "$MODEL_DIR"

echo ""
echo "==> Stage 5: Generate final A/B comparison samples"
.venv/bin/python -m scripts.vocos_finetune.evaluate \
    --checkpoint "$BEST_CKPT" \
    --val-filelist "$DATA_DIR/filelists/val.txt" \
    --output-dir "$TRAINING_DIR/final_comparison" \
    --n-utterances 10

echo ""
echo "============================================"
echo "  Training Complete!"
echo "============================================"
echo ""
echo "Results:"
echo "  MLX weights:     $MODEL_DIR/weights.npz"
echo "  A/B samples:     $TRAINING_DIR/final_comparison/"
echo "  TensorBoard:     tensorboard --logdir $LOG_DIR"
echo ""
echo "Listen to the 4x samples first — clicks are worst there."
echo "Check README.txt in the comparison directory for metrics."

#!/usr/bin/env bash
set -euo pipefail

TRAIN_FILELIST="training/data/filelists/train.txt"
VAL_FILELIST="training/data/filelists/val.txt"
CHECKPOINT_DIR="checkpoints/vocos_phase_reg"
LOG_DIR="logs/vocos_phase_reg"
MAX_STEPS=5000
BATCH_SIZE=16
LR=1e-5
PHASE_COEFF=0.05

RESUME_FLAG=""
if [ "${1:-}" = "--resume" ]; then
    RESUME_FLAG="--resume"
fi

echo "=== Vocos Phase Regularization Training ==="
echo "Phase coeff: $PHASE_COEFF"
echo "LR: $LR"
echo "Max steps: $MAX_STEPS"
echo ""

if [ ! -f "$TRAIN_FILELIST" ]; then
    echo "ERROR: Training filelist not found at $TRAIN_FILELIST"
    echo "Run the vocos_finetune.sh data download first."
    exit 1
fi
echo "[1/3] Data verified."

echo "[2/3] Starting training..."
python -m scripts.vocos_finetune.train_phase_reg \
    --train-filelist "$TRAIN_FILELIST" \
    --val-filelist "$VAL_FILELIST" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log-dir "$LOG_DIR" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --phase-coeff "$PHASE_COEFF" \
    $RESUME_FLAG

echo "[2/3] Training complete."

echo "[3/3] Converting best checkpoint to MLX..."
BEST_CKPT=$(ls -t "$CHECKPOINT_DIR"/*.ckpt | grep -v last | head -1)
python -m scripts.vocos_finetune.convert_mlx \
    --checkpoint "$BEST_CKPT" \
    --output-dir "training/models/vocos-mel-24khz-phase-reg"

echo "=== Done ==="
echo "MLX weights: training/models/vocos-mel-24khz-phase-reg/weights.npz"
echo "TensorBoard: tensorboard --logdir $LOG_DIR"

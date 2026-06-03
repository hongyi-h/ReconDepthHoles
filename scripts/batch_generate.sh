#!/bin/bash
# ============================================================
# Batch Mirror-Scene Generation for Pilot Training Data
# Generates N scenes in parallel using GNU parallel or xargs.
#
# Usage:
#   bash scripts/batch_generate.sh --num_scenes 1000 --num_views 10 --output_dir data/synthetic/train
#   bash scripts/batch_generate.sh --num_scenes 200 --num_views 10 --output_dir data/synthetic/val
#
# Requirements:
#   - Blender >= 5.x in PATH (or set BLENDER_BIN env var)
#   - GNU parallel (optional, falls back to xargs)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BLENDER_SCRIPT="$SCRIPT_DIR/blender_mirror_scene.py"

# Defaults
NUM_SCENES=1000
NUM_VIEWS=10
RESOLUTION="480 640"
OUTPUT_DIR="$PROJECT_ROOT/data/synthetic/train"
JOBS=8  # parallel jobs (set to num CPU cores or less)
BLENDER_BIN="${BLENDER_BIN:-blender}"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_scenes) NUM_SCENES="$2"; shift 2 ;;
        --num_views) NUM_VIEWS="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --blender) BLENDER_BIN="$2"; shift 2 ;;
        --resolution) RESOLUTION="$2 $3"; shift 3 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "Batch Mirror-Scene Generation"
echo "  Scenes: $NUM_SCENES"
echo "  Views per scene: $NUM_VIEWS"
echo "  Resolution: $RESOLUTION"
echo "  Output: $OUTPUT_DIR"
echo "  Parallel jobs: $JOBS"
echo "  Blender: $BLENDER_BIN"
echo "============================================"

# Verify Blender
if ! command -v "$BLENDER_BIN" &>/dev/null; then
    echo "ERROR: Blender not found at '$BLENDER_BIN'"
    echo "Set BLENDER_BIN env var or add blender to PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Generate scene list
generate_scene() {
    local scene_id=$1
    local seed=$scene_id  # deterministic: scene_id = seed
    local scene_dir="$OUTPUT_DIR/scene_$(printf '%05d' $scene_id)"

    # Skip if already generated
    if [[ -f "$scene_dir/scene_meta.json" ]]; then
        return 0
    fi

    "$BLENDER_BIN" --background --python "$BLENDER_SCRIPT" -- \
        --output_dir "$scene_dir" \
        --num_views "$NUM_VIEWS" \
        --resolution $RESOLUTION \
        --seed "$seed" \
        > "$scene_dir.log" 2>&1

    if [[ $? -eq 0 ]]; then
        echo "✓ scene_$(printf '%05d' $scene_id)"
    else
        echo "✗ scene_$(printf '%05d' $scene_id) (see $scene_dir.log)"
    fi
}

export -f generate_scene
export OUTPUT_DIR BLENDER_BIN BLENDER_SCRIPT NUM_VIEWS RESOLUTION

# Run in parallel
if command -v parallel &>/dev/null; then
    echo "Using GNU parallel with $JOBS jobs..."
    seq 0 $((NUM_SCENES - 1)) | parallel -j "$JOBS" generate_scene {}
else
    echo "GNU parallel not found, using xargs (install parallel for better progress)..."
    seq 0 $((NUM_SCENES - 1)) | xargs -P "$JOBS" -I {} bash -c 'generate_scene "$@"' _ {}
fi

# Summary
TOTAL_GENERATED=$(find "$OUTPUT_DIR" -name "scene_meta.json" | wc -l | tr -d ' ')
echo ""
echo "============================================"
echo "Done! Generated $TOTAL_GENERATED / $NUM_SCENES scenes"
echo "Output: $OUTPUT_DIR"
echo "============================================"

# Estimate total size
if [[ $TOTAL_GENERATED -gt 0 ]]; then
    SAMPLE_SIZE=$(du -sh "$OUTPUT_DIR/scene_00000" 2>/dev/null | cut -f1)
    echo "Sample scene size: $SAMPLE_SIZE"
    echo "Estimated total: ~$(echo "$TOTAL_GENERATED" | awk '{printf "%.0f", $1 * 50}') MB (rough)"
fi

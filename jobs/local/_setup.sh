PROJECT_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
STORAGE_ROOT="/mnt/storage"
OUTPUT_DIR="$PROJECT_ROOT/results/jobs/train_adp"
mkdir -p "$OUTPUT_DIR"

MACHINE="LOCAL"

#!/bin/bash
# Recover missing PhysioNet files from S3 using targeted download

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
AWS_CLI="$REPO_ROOT/aws/dist/aws"
S3_BUCKET="s3://physionet-open/challenge-2019/1.0.0/training"
RAW_DIR="$REPO_ROOT/data/raw"
REPORTS_DIR="$REPO_ROOT/data/reports/download"
LOG_FILE="$REPORTS_DIR/recovery.log"

# Ensure directories exist
mkdir -p "$REPORTS_DIR"

# Initialize log
echo "===========================================================" > "$LOG_FILE"
echo "PhysioNet Dataset Recovery" >> "$LOG_FILE"
echo "Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")" >> "$LOG_FILE"
echo "===========================================================" >> "$LOG_FILE"

# Function to download a single file
download_file() {
    local dataset=$1
    local filename=$2
    local s3_path="$S3_BUCKET/$dataset/$filename"
    local local_path="$RAW_DIR/$dataset/$filename"
    
    # Skip if file already exists and is non-empty
    if [ -f "$local_path" ] && [ -s "$local_path" ]; then
        echo "[SKIP] $filename already exists" >> "$LOG_FILE"
        return 0
    fi
    
    # Download with retries
    local max_retries=3
    local retry=0
    
    while [ $retry -lt $max_retries ]; do
        if "$AWS_CLI" s3 cp "$s3_path" "$local_path" --no-sign-request --quiet 2>>"$LOG_FILE"; then
            echo "[OK] Downloaded $filename" >> "$LOG_FILE"
            return 0
        else
            retry=$((retry + 1))
            echo "[RETRY $retry/$max_retries] Failed to download $filename" >> "$LOG_FILE"
            sleep 2
        fi
    done
    
    echo "[FAIL] Failed to download $filename after $max_retries retries" >> "$LOG_FILE"
    return 1
}

# Function to recover files from manifest
recover_from_manifest() {
    local dataset=$1
    local manifest_file=$2
    
    if [ ! -f "$manifest_file" ]; then
        echo "Manifest not found: $manifest_file" >&2
        return 1
    fi
    
    local total=$(wc -l < "$manifest_file")
    local current=0
    local succeeded=0
    local failed=0
    
    echo "Recovering $total missing files for $dataset..."
    echo "Recovering $total files from $manifest_file" >> "$LOG_FILE"
    
    while IFS= read -r filename; do
        current=$((current + 1))
        
        # Progress indicator every 100 files
        if [ $((current % 100)) -eq 0 ]; then
            echo "  Progress: $current/$total files processed..."
        fi
        
        if download_file "$dataset" "$filename"; then
            succeeded=$((succeeded + 1))
        else
            failed=$((failed + 1))
        fi
    done < "$manifest_file"
    
    echo "Recovery complete for $dataset:"
    echo "  Succeeded: $succeeded"
    echo "  Failed: $failed"
    echo ""
    echo "Recovery summary for $dataset:" >> "$LOG_FILE"
    echo "  Total attempted: $total" >> "$LOG_FILE"
    echo "  Succeeded: $succeeded" >> "$LOG_FILE"
    echo "  Failed: $failed" >> "$LOG_FILE"
    
    return $failed
}

# Main recovery process
echo "Starting recovery process..."
echo ""

total_failed=0

# Recover Set A if manifest exists
if [ -f "$REPORTS_DIR/missing_files_setA.txt" ]; then
    echo "Recovering Set A..."
    if ! recover_from_manifest "training_setA" "$REPORTS_DIR/missing_files_setA.txt"; then
        total_failed=$((total_failed + $?))
    fi
fi

# Recover Set B if manifest exists
if [ -f "$REPORTS_DIR/missing_files_setB.txt" ]; then
    echo "Recovering Set B..."
    if ! recover_from_manifest "training_setB" "$REPORTS_DIR/missing_files_setB.txt"; then
        total_failed=$((total_failed + $?))
    fi
fi

echo "===========================================================" >> "$LOG_FILE"
echo "Recovery process complete" >> "$LOG_FILE"
echo "Total failed downloads: $total_failed" >> "$LOG_FILE"
echo "===========================================================" >> "$LOG_FILE"

echo "Recovery process complete."
echo "Log file: $LOG_FILE"
echo "Total failed downloads: $total_failed"

if [ $total_failed -gt 0 ]; then
    echo ""
    echo "WARNING: Some files failed to download. Check log for details."
    exit 1
fi

exit 0

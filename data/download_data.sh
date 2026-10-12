#!/usr/bin/env bash
# data/download_data.sh - Automated acquisition helper for Auto Insurance Fraud Dataset
set -e

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_FILE="${DATA_DIR}/insurance_claims.csv"

if [ -f "${TARGET_FILE}" ]; then
  echo ">>> Dataset already exists at ${TARGET_FILE} ($(wc -l < "${TARGET_FILE}") lines)."
  exit 0
fi

echo ">>> Searching for dataset..."
echo ">>> Attempting download from public GitHub research mirror..."

MIRROR_URL="https://raw.githubusercontent.com/mwitiderrick/insurancedata/master/insurance_claims.csv"

if command -v curl &> /dev/null; then
  curl -fsSL "${MIRROR_URL}" -o "${TARGET_FILE}" || true
elif command -v wget &> /dev/null; then
  wget -q "${MIRROR_URL}" -O "${TARGET_FILE}" || true
fi

if [ -s "${TARGET_FILE}" ]; then
  echo ">>> Successfully downloaded insurance_claims.csv from public research mirror."
  exit 0
fi

echo ">>> Public mirror download unsuccessful or network restricted."
echo ">>> Attempting Kaggle CLI download..."

if command -v kaggle &> /dev/null; then
  kaggle datasets download -d brijbhushannanda1979/auto-insurance-fraud-dataset -p "${DATA_DIR}" --unzip || true
fi

if [ -s "${TARGET_FILE}" ]; then
  echo ">>> Successfully retrieved dataset via Kaggle CLI."
  exit 0
fi

echo ""
echo "=========================================================================="
echo "MANUAL STEP REQUIRED: Data acquisition"
echo "Please obtain insurance_claims.csv from Kaggle or Mendeley Data:"
echo "1. Kaggle: https://www.kaggle.com/datasets/brijbhushannanda1979/auto-insurance-fraud-dataset"
echo "2. Mendeley: https://data.mendeley.com/datasets/992mh7dk9y/2"
echo "Place the downloaded file at: ${TARGET_FILE}"
echo "=========================================================================="
exit 1

#!/usr/bin/env bash
set -euo pipefail

echo "Post-create: Installing Python packages..."
# Upgrade pip, then install requirements (CPU-friendly torch)
python -m pip install --upgrade pip
if [ -f "/workspaces/${PWD##*/}/requirements.txt" ]; then
  pip install -r "/workspaces/${PWD##*/}/requirements.txt"
else
  echo "requirements.txt not found in workspace root — please ensure file exists."
fi

echo "Post-create: Creating data directories..."
mkdir -p /workspaces/"${PWD##*/}"/data/nasa_power
mkdir -p /workspaces/"${PWD##*/}"/data/processed

echo "Post-create: Done."

#!/bin/bash
set -e

for config in configs/NGC300.yaml configs/etg_vlM.yaml configs/etg_lowM.yaml configs/etg_medM.yaml configs/etg_hiM.yaml; do
    echo "=== Running: $config ==="
    python -m sfemulator.data.extract --config "$config"
    echo ""
done

echo "=== All done ==="

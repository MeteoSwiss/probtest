#!/bin/bash
set -e

echo "=== Poetry Lock + Requirements Export ==="

# Step 1: Lock dependencies
echo "Locking Poetry dependencies..."
poetry lock
echo "Dependencies locked successfully."

# Step 2: Export production requirements
echo "Exporting production requirements (requirements.txt)..."
poetry export -f requirements.txt --without dev,test -o requirements.txt
echo "requirements.txt exported successfully."

# Step 3: Export dev & test requirements
echo "Exporting dev & test requirements (requirements_dev.txt)..."
poetry export -f requirements.txt --with dev,test -o requirements_dev.txt
echo "requirements_dev.txt exported successfully."

echo "=== All done! ==="

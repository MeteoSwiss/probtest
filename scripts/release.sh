#!/usr/bin/env bash
set -euo pipefail

# Usage: ./release.sh 1.2.0

NEW_VERSION="$1"
BRANCH="release/v$NEW_VERSION"

echo "Creating release branch $BRANCH..."
git switch --create "$BRANCH"

echo "Updating version in pyproject.toml..."
poetry version "$NEW_VERSION"

echo "Updating dependencies..."
poetry update
./scripts/poetry_lock.sh

echo "Committing changes..."
git add --update
git commit -m "Prepare release v$NEW_VERSION"

echo "Push branch to origin..."
git push -u origin "$BRANCH"

echo "Release branch $BRANCH created and ready for PR."
echo "Next steps:"
echo "1. Open PR against main."
echo "2. Review and merge."
echo "3. Tag release after merge: git tag -a v$NEW_VERSION -m 'Release v$NEW_VERSION'"

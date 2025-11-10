#!/bin/bash
set -e

# Check if uv version command supports --bump (requires uv >= 0.7.0)
if ! uv version --help 2>&1 | grep -q "\-\-bump"; then
    echo "Error: Your version of uv doesn't support 'uv version --bump'"
    echo "Please upgrade to uv >= 0.7.0 or use an alternative tool like bump2version"
    echo ""
    echo "To upgrade uv, run: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Run uv version with all passed arguments and capture the output
output=$(uv version "$@")

# Extract the version from the output
# The output format is "project-name x.y.z => x.y.z" on the last line
# We want the version after the "=>" arrow, or the last version if no arrow
if echo "$output" | grep -q "=>"; then
    version=$(echo "$output" | grep "=>" | tail -1 | sed 's/.*=> *//' | grep -oE '[0-9]+\.[0-9]+\.[0-9]+[a-zA-Z0-9._-]*')
else
    version=$(echo "$output" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+[a-zA-Z0-9._-]*' | tail -1)
fi

if [ -z "$version" ]; then
    echo "Error: Could not extract version from uv output"
    echo "Output was: $output"
    exit 1
fi

echo "Version updated to: $version"

# Create git tag
tag="v$version"
echo "Creating git tag: $tag"
git tag "$tag"

echo "Done! Don't forget to push the tag with: git push origin $tag"

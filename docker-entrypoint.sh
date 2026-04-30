#!/bin/bash
set -e

OLLAMA_BASE="${OLLAMA_HOST:-http://localhost:11434}"
MODEL="${OLLAMA_MODEL:-llama3.2}"

echo ">>> Waiting for Ollama at $OLLAMA_BASE ..."
until curl -sf "$OLLAMA_BASE/api/tags" > /dev/null 2>&1; do
    sleep 3
done
echo ">>> Ollama is ready."

# Pull the model only if it is not already present
if curl -sf "$OLLAMA_BASE/api/tags" | grep -q "\"$MODEL\""; then
    echo ">>> Model '$MODEL' already pulled — skipping download."
else
    echo ">>> Pulling model '$MODEL' (first run — this may take several minutes)..."
    curl -s "$OLLAMA_BASE/api/pull" \
        -H "Content-Type: application/json" \
        -d "{\"name\": \"$MODEL\"}" \
        --no-buffer | grep -v '^$' | tail -5
    echo ">>> Model '$MODEL' ready."
fi

exec "$@"

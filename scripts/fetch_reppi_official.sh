#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMIT="60924986fa23ee53af5500104e5821f16dd3a9e5"
DESTINATION="${REPPI_OFFICIAL_DIR:-$ROOT_DIR/outputs/external/RePPI-$COMMIT}"
REPOSITORY="https://github.com/Wenlong2000/RePPI.git"
REPPI_SHA256="085b702bf8cd4dfa47fe50b0809fd9d7c1cf4865a9e4aac36c2b4a11fb3f8b9c"
PPI_SHA256="85dad473a8026821d9c4cfc58fc615916c83a878d069f85e7bf73304cff85f35"

if [[ ! -d "$DESTINATION/.git" ]]; then
  mkdir -p "$(dirname "$DESTINATION")"
  git clone "$REPOSITORY" "$DESTINATION"
fi

git -C "$DESTINATION" checkout --detach "$COMMIT"

OBSERVED_COMMIT="$(git -C "$DESTINATION" rev-parse HEAD)"
OBSERVED_REPPI_SHA256="$(shasum -a 256 "$DESTINATION/reppi.py" | awk '{print $1}')"
OBSERVED_PPI_SHA256="$(shasum -a 256 "$DESTINATION/ppi.py" | awk '{print $1}')"

if [[ "$OBSERVED_COMMIT" != "$COMMIT" ]]; then
  echo "Official RePPI commit mismatch: $OBSERVED_COMMIT" >&2
  exit 1
fi
if [[ "$OBSERVED_REPPI_SHA256" != "$REPPI_SHA256" ]]; then
  echo "Official reppi.py hash mismatch: $OBSERVED_REPPI_SHA256" >&2
  exit 1
fi
if [[ "$OBSERVED_PPI_SHA256" != "$PPI_SHA256" ]]; then
  echo "Official ppi.py hash mismatch: $OBSERVED_PPI_SHA256" >&2
  exit 1
fi

echo "$DESTINATION"

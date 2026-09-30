#!/usr/bin/env bash
# Fetch and unpack the ChEMBL SQLite release (~5.7 GB download, ~25 GB unpacked).
set -euo pipefail
VERSION="${1:-37}"
DEST="$(dirname "$0")/../data/raw"
mkdir -p "$DEST"
URL="https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_${VERSION}_sqlite.tar.gz"
echo "downloading $URL"
curl -C - -o "$DEST/chembl_${VERSION}_sqlite.tar.gz" "$URL"
echo "unpacking"
tar xzf "$DEST/chembl_${VERSION}_sqlite.tar.gz" -C "$DEST"
find "$DEST" -name "chembl_${VERSION}.db" -print

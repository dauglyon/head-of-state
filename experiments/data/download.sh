#!/usr/bin/env bash
# Downloads the evaluation datasets into ./data (not tracked by git). Run from the repository root.
#   PubMed 20k RCT  (github.com/Franck-Dernoncourt/pubmed-rct): train/dev/test splits, one sentence per line
#   WikiSection     (github.com/sebastianarnold/WikiSection, CC BY-SA 3.0): en_disease and en_city JSON
set -euo pipefail
mkdir -p data/pubmed-rct data/wikisection
for s in train dev test; do
  curl -sfL -o "data/pubmed-rct/$s.txt" \
    "https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/PubMed_20k_RCT/$s.txt"
done
curl -sfL -o data/wikisection/ws.tar.gz \
  "https://github.com/sebastianarnold/WikiSection/raw/master/wikisection_dataset_json.tar.gz"
tar xzf data/wikisection/ws.tar.gz -C data/wikisection LICENSE \
  wikisection_en_disease_train.json wikisection_en_disease_validation.json wikisection_en_disease_test.json \
  wikisection_en_city_train.json
rm data/wikisection/ws.tar.gz
echo "datasets in ./data"

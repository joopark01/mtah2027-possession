#!/bin/sh
# BAR-ATH submission from a fresh clone, one command. Unpack the provided data into data/ first.
#
#     sh reproduce.sh
#
# Builds a venv with the pinned features/requirements.txt, extracts the feature table, decodes it
# with the frozen config (checked against the tracked tune logs in results/tuning/) and writes
# data/derived/submission.csv, validated by the official validator. Prints its sha256 at the end.
# No IDSSE data, tuning or network beyond pip is needed.
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install -q -r features/requirements.txt
.venv/bin/python features/extract_features.py
.venv/bin/python features/make_submission.py
shasum -a 256 data/derived/submission.csv

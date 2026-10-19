#!/bin/sh
cd "$(dirname "$0")"
[ -d venv ] || python3 -m venv venv
. venv/bin/activate
pip install -q -r requirements.txt
# Build the database and train the model once, only if it does not exist yet.
[ -f data/atrisk.db ] || python build_database.py
python app.py

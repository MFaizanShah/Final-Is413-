@echo off
cd /d "%~dp0"
if not exist venv python -m venv venv
call venv\Scripts\activate
rem Installs or updates the packages (quick when everything is already installed).
pip install -q -r requirements.txt
rem Build the database and train the model once, only if it does not exist yet.
if not exist data\atrisk.db python build_database.py
python app.py

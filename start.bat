@echo off
if not exist env\ (
    echo Creating virtual environm...
    python -m venv env
	call .\env\Scripts\activate.bat
	pip install -r requirements.txt
	python app.py
) else (
    call .\env\Scripts\activate.bat
	python app.py
)
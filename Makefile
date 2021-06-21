.PHONY: init test

init: requirements.txt
	pip install -r requirements.txt

test: 
	python -m pytest
	
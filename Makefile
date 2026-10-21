.PHONY: install test docs all

install:
	pip install -r requirements.txt

# Runs the unit tests and writes the HTML test report required by the brief.
test:
	python -m pytest -v --html=reports/test_report.html --self-contained-html

# Generates HTML API documentation from the docstrings.
docs:
	pdoc --docformat google -o docs/api src/porkqc

all: test docs

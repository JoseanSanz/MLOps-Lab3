install:
	pip install uv &&\
	uv sync

test:
	uv run python -m pytest ./tests -vv --cov=main --cov=mylib #test_*.py

format:	
	uv run black mylib/*.py tests/*.py api/*.py cli/*.py

lint:
	uv run pylint --disable=R,C --ignore-patterns=test_.*\.py mylib/model.py api/*.py cli/*.py

refactor: format lint
	
all: install lint test format
install:
	pip install uv &&\
	uv sync

test:
	uv run python -m pytest ./tests -vv --cov=api --cov=cli --cov=mylib

format:	
	uv run black mylib/*.py tests/*.py api/*.py cli/*.py

lint:
	uv run pylint --disable=R,C --ignore-patterns=test_.*\.py mylib/model_onnx_inference.py mylib/train_model_pytorch_oxford_pets.py api/*.py cli/*.py

refactor: format lint
	
all: install lint test format
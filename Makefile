install:
	pip install uv &&\
	uv sync

test:
	uv run python -m pytest ./tests/tests_CICD -vv --cov=api --cov=cli --cov=mylib

test_model:
	uv run python -m pytest ./tests/tests_model -vv

format:	
	uv run black mylib/*.py tests/tests_CICD/*.py tests/tests_model/*.py api/*.py cli/*.py

lint:
	uv run pylint --rcfile=.pylintrc --ignore-patterns=test_.*\.py  mylib/model_onnx_inference.py mylib/train_model_pytorch_oxford_pets.py mylib/select_best_model_and_serialization.py api/*.py cli/*.py

refactor: format lint

train:
	uv run python -m mylib.train_model_pytorch_oxford_pets

serialize:
	uv run python -m mylib.select_best_model_and_serialization
	
all: install format lint test

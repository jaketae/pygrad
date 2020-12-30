.PHONY: quality style test

quality:
	mypy pygrad
	black --check tests pygrad
	isort --check-only tests pygrad
	flake8 tests pygrad

style:
	black --target-version py36 tests pygrad
	isort pygrad

test:
	python -m unittest discover tests
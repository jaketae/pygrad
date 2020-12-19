.PHONY: quality style test

quality:
	black --check --target-version py36 tests pygrad
	isort --check-only tests pygrad
	flake8 --max-line-length 88 tests pygrad

style:
	black --target-version py36 tests pygrad
	isort pygrad

test:
	python -m unittest discover tests
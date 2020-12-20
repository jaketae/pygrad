from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="pygrad",
    version="0.0.2",
    author="Jake Tae",
    author_email="jaesungtae@gmail.com",
    description="A pure Python autograd library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaketae/pygrad",
    packages=find_packages("pygrad", exclude=["tests"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["numpy"],
)

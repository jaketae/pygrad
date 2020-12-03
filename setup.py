import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pygrad",
    version="0.0.1",
    author="Jake Tae",
    author_email="jaesungtae@gmail.com",
    description="A pure Python autograd library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaketae/pygrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

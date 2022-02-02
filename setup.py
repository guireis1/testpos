import setuptools

with open("README_PACKAGE.md", "r") as fh:
    long_description = fh.read()

with open("package_requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="TakeBlipPosTagger",
    version="1.0.6",
    author="Data and Analytics Research",
    author_email="analytics.dar@take.net",
    keywords='postagging',
    description="PosTagger Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True
)

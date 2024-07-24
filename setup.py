from setuptools import setup, find_packages

setup(
    name="irontorch",
    version="0.0.8",
    description="Developer-friendly Pytorch utility library",
    url="https://github.com/thisisiron/irontorch",
    author="Kim Eon",
    author_email="kimiron518@gmail.com",
    license="MIT",
    install_requires=[
        "torch>=1.1",
    ],
    packages=find_packages(),
)

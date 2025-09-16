from setuptools import setup, find_packages

setup(
    name="mag-rg",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "PyQt6",
        "PyQt6-Qt6",
        "PyQt6-sip",
    ],
    entry_points={
        "console_scripts": [
            "rg-annotate=src.rg_ai.annotation_tools.annotate_db:main",
        ],
    },
    author="Maja",
    author_email="mk.maja.kolar@gmail.com",
    description="Rhythmic Gymnastics Annotation Tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mkolar/mag-rg",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 
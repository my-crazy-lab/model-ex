"""
Setup script for Lifelong Learning implementation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lifelong-learning",
    version="0.1.0",
    author="Model-Ex Team",
    description="Comprehensive Lifelong Learning & Continual Learning Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.0.0",
        ],
        "meta": [
            "higher>=0.2.1",
            "learn2learn>=0.1.7",
        ],
        "uncertainty": [
            "torch-uncertainty>=0.1.0",
        ],
        "distributed": [
            "deepspeed>=0.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lifelong-train=examples.text_classification_continual:main",
            "lifelong-eval=training.evaluation:main",
            "lifelong-benchmark=experiments.benchmark:main",
        ],
    },
)

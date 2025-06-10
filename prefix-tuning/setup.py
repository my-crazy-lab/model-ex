"""
Setup script for Prefix Tuning implementation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="prefix-tuning",
    version="0.1.0",
    author="Model-Ex Team",
    description="Comprehensive Prefix Tuning & Prompt Tuning Implementation",
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
        "distributed": [
            "deepspeed>=0.9.0",
            "torch-distributed>=0.1.0",
        ],
        "optimization": [
            "optimum>=1.8.0",
            "onnx>=1.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "prefix-train=examples.text_classification_prefix:main",
            "prompt-train=examples.text_generation_prefix:main",
            "prefix-eval=training.evaluation:main",
        ],
    },
)

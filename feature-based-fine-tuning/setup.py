"""
Setup script for Feature-Based Fine-Tuning package
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="feature-based-fine-tuning",
    version="1.0.0",
    author="AI Research Team",
    author_email="research@example.com",
    description="Comprehensive Feature-Based Fine-Tuning System for Transfer Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/example/feature-based-fine-tuning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
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
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "black>=22.3.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "experiments": [
            "wandb>=0.12.0",
            "optuna>=3.0.0",
            "ray[tune]>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "feature-based-train=training.cli:main",
            "feature-based-eval=evaluation.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.txt"],
    },
    zip_safe=False,
    keywords=[
        "machine learning",
        "deep learning",
        "transfer learning",
        "fine-tuning",
        "feature extraction",
        "transformers",
        "pytorch",
        "nlp",
        "computer vision",
    ],
    project_urls={
        "Bug Reports": "https://github.com/example/feature-based-fine-tuning/issues",
        "Source": "https://github.com/example/feature-based-fine-tuning",
        "Documentation": "https://feature-based-fine-tuning.readthedocs.io/",
    },
)

"""
Setup script for RLHF implementation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rlhf",
    version="0.1.0",
    author="Model-Ex Team",
    description="Comprehensive Reinforcement Learning from Human Feedback Implementation",
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
        "ui": [
            "streamlit>=1.25.0",
            "gradio>=3.35.0",
        ],
        "serving": [
            "fastapi>=0.100.0",
            "uvicorn>=0.22.0",
            "torch-serve>=0.8.0",
        ],
        "distributed": [
            "deepspeed>=0.9.0",
            "torch-distributed>=0.1.0",
        ],
        "safety": [
            "detoxify>=0.5.0",
            "perspective-api>=0.1.0",
        ],
        "evaluation": [
            "mturk>=1.0.0",
            "labelbox>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rlhf-train=examples.chatbot_rlhf:main",
            "rlhf-eval=evaluation.human_eval:main",
            "rlhf-serve=deployment.inference_server:main",
        ],
    },
)

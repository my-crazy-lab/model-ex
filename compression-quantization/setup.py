"""
Setup script for Model Compression & Quantization implementation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="compression-quantization",
    version="0.1.0",
    author="Model-Ex Team",
    description="Comprehensive Model Compression & Quantization Implementation",
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
        "tensorrt": [
            "tensorrt>=8.6.0",
            "pycuda>=2022.1",
        ],
        "mobile": [
            "coremltools>=6.0.0",
            "tensorflow-lite>=2.13.0",
        ],
        "serving": [
            "fastapi>=0.100.0",
            "uvicorn>=0.22.0",
            "onnxruntime-gpu>=1.15.0",
        ],
        "distributed": [
            "deepspeed>=0.9.0",
            "torch-distributed>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "compress-model=examples.bert_quantization:main",
            "quantize-eval=evaluation.benchmarking:main",
            "deploy-model=deployment.onnx_export:main",
        ],
    },
)

"""
Setup script for Cleaning Service Agent

This script handles package installation and dependency management
for the cleaning service booking agent.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Cleaning Service Booking Agent"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="cleaning-service-agent",
    version="1.0.0",
    description="Intelligent conversational agent for cleaning service bookings",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Cleaning Service Team",
    author_email="tech@cleaningservice.com",
    url="https://github.com/cleaningservice/cleaning-service-agent",
    
    # Package configuration
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'pytest-mock>=3.6.0',
            'black>=21.0.0',
            'flake8>=3.9.0',
            'mypy>=0.910'
        ],
        'ml': [
            'scikit-learn>=1.0.0',
            'transformers>=4.0.0',
            'torch>=1.9.0'
        ],
        'database': [
            'sqlalchemy>=1.4.0',
            'psycopg2-binary>=2.9.0'
        ],
        'api': [
            'requests>=2.25.0',
            'httpx>=0.24.0'
        ],
        'monitoring': [
            'prometheus-client>=0.11.0',
            'structlog>=21.1.0'
        ]
    },
    
    # Package data
    include_package_data=True,
    package_data={
        'cleaning-service-agent': [
            'data/*.json',
            'config/*.yaml',
            'config/*.json'
        ]
    },
    
    # Entry points
    entry_points={
        'console_scripts': [
            'cleaning-agent=examples.basic_conversation:main',
            'cleaning-demo=examples.basic_conversation:interactive_demo'
        ]
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Communications :: Chat",
        "Topic :: Office/Business :: Scheduling"
    ],
    
    # Keywords
    keywords="chatbot, conversational-ai, booking-system, cleaning-service, nlp",
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/cleaningservice/cleaning-service-agent/issues",
        "Source": "https://github.com/cleaningservice/cleaning-service-agent",
        "Documentation": "https://cleaning-service-agent.readthedocs.io/"
    }
)

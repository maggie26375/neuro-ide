"""
Setup script for SE+ST Combined Model Package
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "SE+ST Combined Model for Cross-Cell-Type Perturbation Prediction"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="se-st-combined",
    version="0.1.0",
    author="maggie26375",
    author_email="your-email@example.com",
    description="SE+ST Combined Model for Cross-Cell-Type Perturbation Prediction",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/maggie26375/se-st-combined",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [
            "se-st-train=se_st_combined.cli.train:main",
            "se-st-infer=se_st_combined.cli.infer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "se_st_combined": [
            "configs/*.yaml",
            "data/*",
        ],
    },
    zip_safe=False,
)

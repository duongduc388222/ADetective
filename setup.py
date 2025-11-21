from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="adetective",
    version="0.1.0",
    author="Research Team",
    description="Oligodendrocyte AD Pathology Classifier using single-cell RNA-seq",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "scgpt": ["scgpt>=0.2.0", "flash-attn==2.6.3"],
        "dev": ["pytest>=7.0", "black>=23.0", "isort>=5.0", "flake8>=6.0"],
    },
)

from setuptools import find_packages, setup
import pathlib

setup(
    name="olaf",
    version=pathlib.Path("olaf/VERSION").read_text(encoding="utf-8"),
    description="A deep learning library built from scratch.",
    long_description=pathlib.Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/neluca/olaf",
    author="Neluca",
    author_email="myneluca@gmail.com",
    license="MIT",
    project_urls={
        "Source Code": "https://github.com/neluca/olaf",
        "Issues": "https://github.com/neluca/olaf/issues",
    },
    classifiers=[
        "Development Status :: 1 - Beta",
        "Environment :: GPU :: NVIDIA CUDA :: 11",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.12",
        "Topic :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.12",
    install_requires=["numpy>=1.26.4"],
    extras_require={
        "dev": [
            "mypy>=1.11.2",
            "pytest>=8.2.0",
            "pytest-cov>=5.0.0",
            "torch>=2.5.0",
            "twine>=5.1.1",
            "wheel>=0.43.0",
        ],
    },
    packages=find_packages(exclude=["tests", ".github", ".venv", "docs"]),
    include_package_data=True,
)

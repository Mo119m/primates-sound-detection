from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="primate-vocalization-detection",
    version="1.0.0",
    author="Moshi Fu",
    author_email="mfu39@wisc.edu",
    description="Automated detection of primate vocalizations in rainforest audio recordings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mo119m/primates-sound-detection",
    package_dir={"primate_detection": "src"},
    packages=["primate_detection"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    # No console script is exposed yet – ``detection.py`` does not define a
    # ``main`` entry point, so adding one here would make ``pip install`` fail.
)

from setuptools import setup, find_packages

setup(
    name="visdetect_photom",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
        "pyyaml",
    ],
)

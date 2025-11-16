from setuptools import setup, find_packages

setup(
    name="nlp_prj",
    version="0.1",
    description="nlp_prj",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)

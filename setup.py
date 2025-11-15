from setuptools import setup, find_packages

setup(
    name="dss",
    version="0.1",
    description="dss",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)

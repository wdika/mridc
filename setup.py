from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mridc",
    version="0.0.1",
    packages=["mridc", "mridc.nn", "mridc.nn.rim", "mridc.data", "tests", "tests.fastmri", "tools", "scripts"],
    url="https://github.com/wdika/mridc",
    license="Apache-2.0 License ",
    author="Dimitrios Karkalousos",
    author_email="d.karkalousos@amsterdamumc.nl",
    description="Data Consistency for Magnetic Resonance Imaging",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
)

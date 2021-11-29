from setuptools import setup

setup(
    name="mridc",
    version="v.0.0.1",
    packages=["mridc", "mridc.nn", "mridc.nn.rim", "mridc.data", "tests", "tests.fastmri", "tools", "scripts"],
    url="https://github.com/wdika/mridc",
    license="Apache-2.0 License ",
    author="Dimitrios Karkalousos",
    author_email="d.karkalousos@amsterdamumc.nl",
    description="Data Consistency for Magnetic Resonance Imaging",
)

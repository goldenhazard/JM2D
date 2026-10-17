from setuptools import find_packages, setup

setup(
    name="jm2d_d3il",
    version="1.0.0",
    description="JM2D on D3IL-Avoiding (arXiv 2509.08775, Table 2 / Table 3): DPCC, SafeDiffuser, MPD and JM2D planners",
    packages=find_packages(include=["jm2d_d3il", "jm2d_d3il.*", "diffuser", "diffuser.*"]),
    python_requires=">=3.10",
)

from setuptools import find_namespace_packages, setup

setup(
    name="jm2d_maze",
    version="1.0.0",
    description="JM2D on D4RL PointMaze (arXiv 2509.08775, Sec. 5.1): DP / RAIL / Gibbs / JM2D safety filtering",
    packages=find_namespace_packages(include=["jm2d_maze", "jm2d_maze.*", "safediffusion", "safediffusion.*",
                                              "robomimic", "robomimic.*"]),
    package_data={"jm2d_maze": ["reachability/frs_tensor_saved/*.mat"]},
    python_requires=">=3.8",
)

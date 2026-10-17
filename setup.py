from setuptools import find_packages, setup

setup(
    name="jm2d_toy",
    version="1.0.0",
    description="JM2D toy domain (arXiv 2509.08775): joint sampling (Fig. 2) and constrained generation (Fig. 9)",
    packages=find_packages(include=["jm2d_toy", "jm2d_toy.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1",
        "numpy",
        "scipy",
        "diffusers>=0.31",
        "einops",
        "tqdm",
        "matplotlib",
    ],
)

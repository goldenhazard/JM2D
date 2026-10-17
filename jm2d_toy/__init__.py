"""JM2D toy domain (arXiv 2509.08775): tasks ``donut_joint`` (joint sampling, paper Fig. 2) and
``donut_constrained`` (constrained generation, paper Fig. 9).

Sub-packages:
    jm2d_toy.samplers  sampler registry: ``make`` (donut_constrained) and ``make_joint`` (donut_joint)
    jm2d_toy.tasks     problems: ``DonutConstrainedProblem`` and ``DonutJointProblem``
"""
import os

# must be set before the CUDA context exists for bitwise-reproducible cuBLAS results
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

__version__ = "1.0.0"

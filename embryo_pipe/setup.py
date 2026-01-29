from setuptools import setup, find_packages

setup(
    name="embryo_pipe",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy", "scikit-image", "cellpose", "czifile", "pandas", "seaborn"
    ],
)
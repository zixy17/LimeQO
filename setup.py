from setuptools import setup, find_packages

setup(
    name="limeqo", 
    version="0.1.0",
    description="A framework for low rank offline query optimization",
    author="Zixuan Yi",
    author_email="zixy@seas.upenn.edu",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "torch", 
        "scikit-learn",
        "tqdm",
    ],
)

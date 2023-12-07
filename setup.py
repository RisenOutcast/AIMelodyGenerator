from setuptools import setup, find_packages

setup(
    name="TekoalyMusic",
    version="1.0.0",
    description="Program for generating music",
    author="Metso",
    author_email="risenoutcast@gmail.com",
    packages=find_packages(),
    install_requires=[
        "pretty_midi==0.2.10",
        "numpy==1.26.2",
        "tensorflow==2.15.0",
        "IPython==8.18.1",
        "matplotlib==3.8.2",
        "pandas==2.1.3",
        "seaborn==0.13.0"
    ],
)

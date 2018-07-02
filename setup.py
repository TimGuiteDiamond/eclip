from setuptools import setup

setup(
    name = "eclip",
    version = "0.0.1",
    author = "Jenna Elliott",
    author_email = "jenna.elliott@diamond.ac.uk",
    description = "A package that classifies maps",
    license = "BSD",
    keywords = "awesome python package",
    packages=[
      'eclip', 
      'tests'
    ],
    scripts=[
    ],
    install_requires=[
      'pytest',
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
    ],
)

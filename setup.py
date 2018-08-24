from setuptools import setup

setup(
    name = "eclip",
    version = "0.0.1",
    author = "Jenna Elliott",
    author_email = "jenna.elliott@diamond.ac.uk",
    description = "A proof of principle package that applies machine learning to classify electron density maps",
    license = "BSD",
    keywords = "electron density classification",
    packages=[
      'eclip', 
      'tests'
    ],
    scripts=[
      'bin/RunTrain',
      'bin/RunPred',
      'bin/ConvMAP',
      'bin/learn',
      'bin/EP_success',
      'bin/predic',
      'bin/predictest'
    ],
    install_requires=[
      'pytest',
      'keras',
      'tensorflow',
      'matplotlib',
      'scikit-learn',
      'Pillow',
      'numpy',
      'scipy',
      'mrcfile',
      'logging',
      'argparse'
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
    ],
)

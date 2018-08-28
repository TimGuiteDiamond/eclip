# Eclip

Elip is a package that trains a CNN model to recognise good and bad electron
density maps. 

Eclip reads in the electron density maps in .map format and slices them into 2D
images. These images are then used to train a model that will be able to give a
label to each image. These labels are then averaged to give an overall score for
the map. 

The package is split into two sections. The first is used to train the model,
the second is used to implement this model on new data. 

## Getting Started 

These instructions will get a copy of the project working on your local
machine.

### Prerequisites

To install this software you will need: 

* keras
* tensorflow
* matplotlib
* scikit-learn
* Pillow
* numpy
* scipy
* mrcfile
* logging
* argparse

These should be installed when the environment is setup, however, sometimes
there are issues when installing and using tensorflow and keras. 

For the program to work the map files need to be stored in one directory.
This layout is automatically created by the Topaz program if being used before
Eclip. 

The SQLite database must have a table called "**Phasing**", with collumn names: 
* pdb\_id\_id
* ep\_success\_o
* ep\_percent\_o
* ep\_score\_o
* ep\_score\_rawo
* ep\_confidence\_o
* ep\_confidene\_rawo
* ep\_success\_i
* ep\_percent\_i
* ep\_score\_i
* ep\_score\_rawi
* ep\_confidence\_i
* ep\_confidence\_rawi
* ep\_img\_o
* ep\_raw\_o
* ep\_img\_i
* ep\_raw\_i

### Installing 

To get a development env running:

At the moment, this is how the environment is setup. 
Download various files.
Go to directory containing setup.py and type:
```
> python3 setup.py develop --user

> export PATH=$PATH:~/.local/bin
```

The functions should then be available to use in command line. 

### How to use

To use Eclip, it is simplest to run first the training then implementing
programs, however each individual section can be run independantly. Running
independantly allows for more flexibility in the arguments.  

The sections are as follows: 
* **ConvMAP:** Converts the maps into images
* **EP\_success** Updates an SQLite database
* **learn** trains a new model
* **predictest** tests the predictions of the new model
* **predic** uses a pretrained model on new images

**RunTrain** is comprised of ConvMAP, EP\_success, learn and predictest

**RunPred** is comprised of ConvMAP and predic

RunTrain  is run by typing "RunTrain" in the command line. 

The arguments required are as follows:
* --input: The imput map directory
* --output: The output directory for the images
* --db: The location of the sqlite database file
* --raw: Boolean, True if data is just for heavy atom positions
* --epochs: Number of epochs to train for
* --nmb: Number of images to train on per axis per protein
* --trial: The starting trial number
* --lgfir: The location of the output directory for stats

Example of how to call: 
```
>RunTrain --epochs=100 --nmb=15
```

RunPred is similarly run by typing "RunPred" in the command line.

The arguments required are as follows:
* --input: The input map directory
* --output: The output directory for the images
* --db: The locaiton for the sqlite database file
* --raw: Boolean, True if data is just for heavy atom positions
* --op: output location for prediciton file

Examples of how to call:
```
>RunPred --input=/INPUT-IMAGE-DIRECTORY/ --output=/OUTPUT-IMAGE-DIRECTORY
```

### Running tests

To run the tests...

'''
example of tests
'''

expected outcome
'''
expected outcome
'''

## Authors

* **Jenna Elliott** 

## Acknowledgements

* **Melanie Vollmar**: Supervisor
* **James Parkhurst**
* **Gwyndaf Evans**


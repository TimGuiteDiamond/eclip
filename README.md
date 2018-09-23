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

#### Software
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

Python3 must also be installed inorder to run this program. 

#### Data
For the program to work the map files need to be stored in one directory.
This layout is automatically created by the Topaz program if being used before
Eclip. 

If using RunTrain or EP\_success, there must also be a directory with a
sub-directory for each of the protein names. Each of these sub diectories must
have a log file (each one with the same name). These log files (default is
simple\_xia2\_to\_shelxcde.log) need to contain the characters: "Best space group:
" followed by a space group. 
In the same sub-direcotry there must also be a directory names by this space
group. In there there must be a .lst file with the same name as the protein
(with \_i if inverse phasing). This file must contain the characters "with CC "
followed by a percentages, between the last 1000 and last 700 characters.
Alternatively, it can contain the words "CC is less than zero - giving up". This
is so that EP\_success can assign scores to the proteins for training.

A demonstration of this: 

  - MapDirectory
      - ProteinName1.map
      - ProteinName1_i.map
      - ProteinName2.map
      .
      .
      .


  - EP\_phasing
      - ProteinName1
          - SpaceGroup
              - ProteinName1.lst
          'simple_xia2_to_shelxcde.log'
      - ProteinName2
      .
      .
      .
    

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

To get a development env running, first download the package. Then go to the
direcotry containing setup.py and type the following into the command line: 

```
> python3 setup.py develop --user

> export PATH=$PATH:~/.local/bin
```

The functions should then be available to use in command line. 

### How to use

To use Eclip, it is simplest to run first the training function (RunTrain), and then the predicting
function (RunPred). However, each individual section can also be run independantly. This allows for 
more flexibility in the arguments.  

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

### Pretrained Models

There are two pretrained models in this package. Their performance and data type
is found in the table below. 

| Model | Data | Maps | Images | Training | Predicting | Image Accuracy | Map Accuracy |
|-------|--|---|---|--|--|--|--|
| 1 | Feature Enhanced | 786 | 35370 | 5h49min | 34sec | 97% | 93% |
| 2 | Heavy Atoms | 791 | 791 | 35595 | 4h17min | | 81% | 76% |
| 3 | Initial Phases | 845 | 38025 | 5h32min | | 86% | 70% |

The total number of maps and images used in the creation of these models (the
sum of those used for training and predicting are in the columns 'maps and
'Images' respectively. The runtimes are under 'Training' and 'Predicting'. the
training runtime was calculated for the program running on a 4 Nvidia Tesla GPU
cluster, and the predicting runtime was calculated for a single MGA-G200e GPU.

Details on the data types and how they were produced can be found in the
doccumentation. In summary, the Feature Enhanced Data is data for the electron
density of the structure, having been improved using SHELXe, whereas the Heavy
Atom Positions data includes only the information on the positions of the heavy
atoms in the structure. The Initial Phases data is data for the electron density
of the structure, but with no feature enhancement. 

### Doccumentation

The doccumentation can be found in the directory /Documentation/html directory.
Once the package is downloaded, open the index.html file in Google chrome or
firefox to access the doccumentation. There is also a pdf copy available under
/Documentation/pdf, however this lacks the interactive links of the html
version. 

## Authors

* **Jenna Elliott** 

## Acknowledgements

* Supervisor: **Melanie Vollmar**:
* Co-Supervisor: **James Parkhurst**
* Principle Investigator: **Gwyndaf Evans**


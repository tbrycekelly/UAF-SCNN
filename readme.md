# UAF-SCNN: a SparseConvNet implementation

## About this software


## Setup

For the first install,

    cd /opt
    sudo git clone https://github.com/tbrycekelly/UAF-SCNN.git
    sudo chown -R plankline:plankline /opt/UAF-SCNN
    sudo chmod 776 -R /opt/UAF-SCNN

To update from github,

    cd /opt/UAF-SCNN
    git clean -f
    git pull

May need to run `git config --global --add safe.directory /opt/UAF-SCNN` if permissions are not right.

To build SCNN:

    cd /opt/UAF-SCNN/build
    make clean
    make wp2

If there are undefined refences on the make, check that the LIBS line in ./build/Makefile is the output of `pkg-config opencv --cflags --libs` and inclues `-lcublas`.

Test it and copy it to final directory (if it works):

    ./wp2
    cp ./wp2 ../scnn


We build the `./scnn` executable from the wp2.cpp source code contained in the build subfolder. This should be the starting point for any soruce code edits or modifications to the existing executable. 

#### Quickstart

__Options__

Command line arguments for `./scnn`:

    -start NUM  [400]
    -stop NUM   [400]
    -batchSize NUM  [350]
    -train DIR  [Data/plankton/train]
    -unl DIR    [data/plankton/test]
    -nClasses NUM   [-1]
    -exemplarsPerClassPerEpoch NUM  [1000]
    -initialLearningRate NUM    [0.003]
    -learningRateDecay NUM  [0.01]
    -validationSetPercentage NUM    [0]
    -cudaDevice NUM
    -basename STR   [plankton]


Example command line call (taken from an actual segmentation.py log file):

    ./scnn -start 324 -stop 324 -unl /tmp/segment/Camera3_VIPF-306-2022-07-21-22-36-51.647 -cD 1

#### Training a Neural Network

Copy the training dataset into /opt/UAF-SCNN/Data/plankton/train so that images are in subfolders by category, e.g.: /opt/UAF-SCNN/Data/plankton/train/detritus/iamge.jpg

Run `classList.sh`

    cd /opt/UAF-SCNN/Data/plankton
    ./classList.sh

You may wish to change the minimum sample size required for a taxa to be included by modifying the _minN_ value within classList.sh. Taxa folders with fewer than _minN_ images will not be included in the training.

_TODO: Complete this section_



## Licensing and Use
SparseConvNet is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

SparseConvNet is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.


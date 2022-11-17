#!/bin/bash

# file to store the list of classes
listFile="classList"
# minimum number of images in a class
minN=10

rm -f $listFile

for item in $(ls train);
do
  # for each directory
  if [ -d "train/$item" ]; then
    # count the number of images
    n=$(ls -1 train/$item | wc -l)
    if [ "$n" -gt $minN ]; then
      # store in file
      echo $n $item
      echo $item >> $listFile
    else
      # just print
      echo $n $item "XX"
    fi
  fi
done

exit 0

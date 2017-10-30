#!/bin/bash

#simple script for resizing images in all class directories
#also reformats everything from whatever to png

S_PATH=$(cd $(dirname $0)/../../..;pwd)
DATA_PATH=${S_PATH}/data
testimages=${DATA_PATH}/test-images
trainingimages=${DATA_PATH}/training-images


if [ `ls ${testimages}/*/*.jpg 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in ${testimages}/*/*.jpg; do
    convert "$file" -resize 28x28\! "${file%.*}.png"
    file "$file" #uncomment for testing
    rm "$file"
  done
fi

if [ `ls ${testimages}/*/*.png 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in ${testimages}/*/*.png; do
    convert "$file" -resize 28x28\! "${file%.*}.png"
    file "$file" #uncomment for testing
  done
fi

if [ `ls ${trainingimages}/*/*.jpg 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in ${trainingimages}/*/*.jpg; do
    convert "$file" -resize 28x28\! "${file%.*}.png"
    file "$file" #uncomment for testing
    rm "$file"
  done
fi

if [ `ls ${trainingimages}/*/*.png 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in ${trainingimages}/*/*.png; do
    convert "$file" -resize 28x28\! "${file%.*}.png"
    file "$file" #uncomment for testing
  done
fi

#!/bin/bash

if [ ! -f "focalplane.txt" ]; then
    curl -SL https://www.dropbox.com/s/x4j5ke5iq2ngo0w/focalplane.txt?dl=1 -o focalplane.txt
fi

if [ ! -f "boresight.txt" ]; then
    curl -SL https://www.dropbox.com/s/5pckf9qlf8h4si7/boresight.txt?dl=1 -o boresight.txt
fi

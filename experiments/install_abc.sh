#! /bin/bash

echo "Clone and compile abc" 
git clone https://github.com/berkeley-abc/abc
cd abc 
git reset --hard 9478c17
make -j4 abc



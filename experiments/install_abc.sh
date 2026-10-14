#! /bin/bash
set -e

echo "Clone and compile abc"
git clone https://github.com/berkeley-abc/abc
cd abc
git checkout --detach 9478c17
make -j4 abc

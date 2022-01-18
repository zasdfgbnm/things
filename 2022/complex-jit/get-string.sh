#!/bin/bash

echo "#include <complex>" | clang++ -x c++ -stdlib=libc++ -E -P - -o complex-standalone.cpp

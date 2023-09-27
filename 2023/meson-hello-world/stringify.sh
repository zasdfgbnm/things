#!/bin/bash

echo "const char* hello = \"$(cat $1)\";" > $2

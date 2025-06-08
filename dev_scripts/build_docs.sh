#!/bin/bash
# This script purges the docs and rebuilds them

cd ../docs
/bin/rm -r build
/bin/rm -r source/generated

make clean html

echo ""
echo "*** The html documentation is at mbirjax/docs/build/html/index.html ***"
echo ""

cd ../dev_scripts

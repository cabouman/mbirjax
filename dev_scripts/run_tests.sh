#!/bin/bash

echo "Running pytest with multiple workers on all tests."

pytest -n 10 ../tests



#!/bin/bash

path="../../../data/safebooru/upper_body_768"

python simplify.py --img "${path}/sketch/enhanced" --out "${path}/line/enhanced" || exit 1

python simplify.py --img "${path}/sketch/original" --out "${path}/line/original" || exit 1

python simplify.py --img "${path}/sketch/pured" --out "${path}/line/pured" || exit 1

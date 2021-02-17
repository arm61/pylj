#!/usr/bin/env bash
set -e

CWD=$(pwd -P)
docker run \
    -u $(id -u):$(id -g) -v ${CWD}:${CWD} -w ${CWD} -e HOME=${CWD} \
    -it --rm -p 8888:8888 pythoninchemistry:latest

#!/usr/bin/env bash
set -euo pipefail

flatc \
    --go \
    --gen-all \
    --go-module-name github.com/maxpoletaev/go-catboost \
    --go-namespace cbm \
    -I flatbuffers \
    -o internal \
    flatbuffers/catboost/libs/model/flatbuffers/model.fbs

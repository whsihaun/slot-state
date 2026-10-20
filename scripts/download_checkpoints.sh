#!/usr/bin/env bash

curl -L -o checkpoints.tar.gz https://github.com/whsihaun/slot-state/releases/download/v1.0/slot-state-checkpoints-v1.0.tar.gz
echo "b783d224459e181ed6dcfe3f857f3c7ee4cb8877e3dba7d3faf097d4c8a59348  checkpoints.tar.gz" | sha256sum -c
tar -xzf checkpoints.tar.gz
rm checkpoints.tar.gz

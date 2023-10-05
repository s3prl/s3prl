#!/usr/bin/env bash
set -euo pipefail

if [ $# != 0 ]; then
    echo "Usage: $0"
    exit 1;
fi

if [ ! -e espnet.done ]; then
    if ! python3 -c "import espnet2" &> /dev/null; then
        pip install "espnet>=202308"
    else
        echo echo "espnet is already installed"
    fi
    touch espnet.done
else
    echo "espnet is already installed"
fi



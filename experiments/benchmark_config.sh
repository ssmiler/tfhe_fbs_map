#!/usr/bin/env bash
# Source this file to configure benchmark trials and randomization.

NB_TRIALS=${NB_TRIALS:-1}
RANDOM_NODE_ORDER=${RANDOM_NODE_ORDER:-0}
RANDOM_TIE_BREAKING=${RANDOM_TIE_BREAKING:-0}

[[ $NB_TRIALS =~ ^0*[1-9][0-9]*$ ]] || {
    echo "NB_TRIALS must be a positive integer" >&2
    return 1
}
[[ $RANDOM_NODE_ORDER =~ ^[01]$ && $RANDOM_TIE_BREAKING =~ ^[01]$ ]] || {
    echo "RANDOM_NODE_ORDER and RANDOM_TIE_BREAKING must be 0 or 1" >&2
    return 1
}
export NB_TRIALS RANDOM_NODE_ORDER RANDOM_TIE_BREAKING

MAPPER_FLAGS="--trials $NB_TRIALS"
if [[ $RANDOM_NODE_ORDER == 1 ]]; then
    MAPPER_FLAGS+=" --random-node-order"
fi
if [[ $RANDOM_TIE_BREAKING == 1 ]]; then
    MAPPER_FLAGS+=" --random-tie-breaking"
fi

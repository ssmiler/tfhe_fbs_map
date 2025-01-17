#! /bin/bash

# clone and compile abc
[ -f abc/abc ] || ( echo "Clone and compile abc" && git clone https://github.com/berkeley-abc/abc && cd abc && make -j4 abc ) || exit

# clone EPFL benchmarks
[ -d benchmarks/epfl ] || ( echo "Clone EPFL circuit benchmark" && git clone https://github.com/lsils/benchmarks benchmarks/epfl ) || exit

BENCHES=$(ls benchmarks/epfl/arithmetic/*.blif)
BENCHES+=" "$(ls benchmarks/epfl/random_control/*.blif)

FBS_SIZES=$(seq 3 16)

MAP_CIRCUIT_PY="../fbs_mapper/map_circuit.py"

OUTPUT_DIR=outputs/epfl

rm -f Makefile
ALL=""

# show current Makefile
echo "log_makefile:" >> Makefile
echo -e "\t@cat Makefile" >> Makefile
echo -e "\t@echo" >> Makefile
echo -e "\t@echo" >> Makefile
echo >> Makefile

# targets for mapping bench circuits to FBSs
echo "$OUTPUT_DIR:" >> Makefile
echo -e "\t@mkdir -p $OUTPUT_DIR" >> Makefile
echo >> Makefile
ALL+=" $OUTPUT_DIR"

function run_bench() {
    BLIF=$1
    BENCH=$2
    FBS_SIZE=$3
    MAPPER=$4

    OUT="$OUTPUT_DIR/$BENCH"_"$FBS_SIZE"_"$MAPPER.fbs"
    OUT_LBF="$OUTPUT_DIR/$BENCH"_"$FBS_SIZE"_"$MAPPER.lbf"
    LOG="$OUTPUT_DIR/$BENCH"_"$FBS_SIZE"_"$MAPPER.log"
    ALL+=" ${OUT_LBF}"

    echo "$OUT $OUT_LBF $LOG: $BLIF | $OUTPUT_DIR"
    echo -e "\tpython3 $MAP_CIRCUIT_PY $BLIF --fbs_size $FBS_SIZE --mapper $MAPPER --output $OUT --output_lbf ${OUT_LBF} > $LOG 2>&1"
    echo ""
}

for BLIF in $BENCHES
do
    BENCH=$(basename -- "$BLIF" .blif)

    for MAPPER in "basic" "search"
    do
        run_bench $BLIF $BENCH 2 $MAPPER >> Makefile
    done

    for FBS_SIZE in $FBS_SIZES
    do
        for MAPPER in "naive" "search"
        do
            run_bench $BLIF $BENCH $FBS_SIZE $MAPPER >> Makefile
        done
    done

done

echo ".DEFAULT_GOAL := all" >> Makefile
echo "all: $ALL" >> Makefile

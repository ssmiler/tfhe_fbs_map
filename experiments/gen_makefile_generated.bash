#! /bin/bash

# clone and compile abc
[ -f abc/abc ] || ( echo "Clone and compile abc" && git clone https://github.com/berkeley-abc/abc && cd abc && make -j4 abc ) || exit

# generate benchmarks
[ -d benchmarks/generated ] || ( echo "Generating benchmarks" && mkdir -p benchmarks/generated && python3 generate_benchmarks.py --prefix benchmarks/generated ) || exit

BENCHES=$(ls benchmarks/generated/*.blif)

FBS_SIZES=$(seq 3 16)

MAP_CIRCUIT_PY="../fbs_mapper/map_circuit.py"

MAPPERS="naive search"

OUTPUT_DIR=outputs/generated

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

for BLIF in $BENCHES
do
    BENCH=$(basename -- "$BLIF" .blif)

    FBS_SIZE=2
    for MAPPER in "basic" "search"
    do
        OUT="$OUTPUT_DIR/$BENCH"_"$FBS_SIZE"_"$MAPPER.fbs"
        LOG="$OUTPUT_DIR/$BENCH"_"$FBS_SIZE"_"$MAPPER.log"
        echo "$OUT $LOG: $BLIF | $OUTPUT_DIR" >> Makefile
        echo -e "\tpython3 $MAP_CIRCUIT_PY $BLIF --fbs_size $FBS_SIZE --mapper $MAPPER --output $OUT > $LOG 2>&1" >> Makefile
        echo >> Makefile
        ALL+=" $OUT"
    done

    for FBS_SIZE in $FBS_SIZES
    do
        for MAPPER in $MAPPERS
        do
            OUT="$OUTPUT_DIR/$BENCH"_"$FBS_SIZE"_"$MAPPER.fbs"
            LOG="$OUTPUT_DIR/$BENCH"_"$FBS_SIZE"_"$MAPPER.log"
            echo "$OUT $LOG: $BLIF | $OUTPUT_DIR" >> Makefile
            echo -e "\tpython3 $MAP_CIRCUIT_PY $BLIF --fbs_size $FBS_SIZE --mapper $MAPPER --output $OUT > $LOG 2>&1" >> Makefile
            echo >> Makefile
            ALL+=" $OUT"
        done
    done

done

echo ".DEFAULT_GOAL := all" >> Makefile
echo "all: $ALL" >> Makefile

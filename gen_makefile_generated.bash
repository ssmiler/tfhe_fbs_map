#! /bin/bash

# clone and compile abc
[ -f abc/abc ] || ( echo "Clone and compile abc" && git clone https://github.com/berkeley-abc/abc && cd abc && make -j4 abc ) || exit

# generate benchmarks
[ -d benchmarks/generated ] || ( echo "Generating benchmarks" && mkdir -p benchmarks/generated && python3 generate_benchmarks.py --prefix benchmarks/generated ) || exit

BENCHES=$(ls benchmarks/generated/*.blif)

# FBS_SIZES="4 8 16 32 64"
FBS_SIZES=$(seq 2 16)

MAPPERS="naive search"

BENCH_XAG_DIR=outputs/benchmarks_xag/generated
OUTPUT_DIR=outputs/generated

rm -f Makefile
ALL=""

# show current Makefile
echo "log_makefile:" >> Makefile
echo -e "\t@cat Makefile" >> Makefile
echo -e "\t@echo" >> Makefile
echo -e "\t@echo" >> Makefile
echo >> Makefile

# targets for mapping bench circuits to XAGs
echo "$BENCH_XAG_DIR:" >> Makefile
echo -e "\t@mkdir -p $BENCH_XAG_DIR" >> Makefile
echo >> Makefile
ALL+=" $BENCH_XAG_DIR"

BLIFS=""
for BENCH in $BENCHES
do
    BLIFS+=" $BENCH"

    # BENCH_BASE=$(basename -- "$BENCH" .blif)
    # BLIF_XAG="$BENCH_XAG_DIR/${BENCH_BASE}-xag.blif"

    # echo "$BLIF_XAG: $BENCH | $BENCH_XAG_DIR" >> Makefile
    # echo -e "\t./abc/abc -c \"read_blif $BENCH; read_library lib.genlib; ps; map; ps; unmap; ps; write_blif $BLIF_XAG\"" >> Makefile
    # echo >> Makefile

    # BLIFS+=" $BLIF_XAG"
done
ALL+=" $BLIFS"

# targets for mapping bench circuits to FBSs
echo "$OUTPUT_DIR:" >> Makefile
echo -e "\t@mkdir -p $OUTPUT_DIR" >> Makefile
echo >> Makefile
ALL+=" $OUTPUT_DIR"

for BLIF in $BLIFS
do
    BENCH=$(basename -- "$BLIF" .blif)

    MAPPER=basic
    FBS_SIZE=2
    OUT="$OUTPUT_DIR/$BENCH"_"$FBS_SIZE"_"$MAPPER.fbs"
    LOG="$OUTPUT_DIR/$BENCH"_"$FBS_SIZE"_"$MAPPER.log"
    echo "$OUT $LOG: $BLIF | $OUTPUT_DIR" >> Makefile
    echo -e "\tpython3 map_circuit.py $BLIF --fbs_size $FBS_SIZE --mapper $MAPPER --output $OUT > $LOG 2>&1" >> Makefile
    echo >> Makefile
    ALL+=" $OUT"

    for FBS_SIZE in $FBS_SIZES
    do
        for MAPPER in $MAPPERS
        do
            OUT="$OUTPUT_DIR/$BENCH"_"$FBS_SIZE"_"$MAPPER.fbs"
            LOG="$OUTPUT_DIR/$BENCH"_"$FBS_SIZE"_"$MAPPER.log"
            echo "$OUT $LOG: $BLIF | $OUTPUT_DIR" >> Makefile
            echo -e "\tpython3 map_circuit.py $BLIF --fbs_size $FBS_SIZE --mapper $MAPPER --output $OUT > $LOG 2>&1" >> Makefile
            echo >> Makefile
            ALL+=" $OUT"
        done
    done

done

echo ".DEFAULT_GOAL := all" >> Makefile
echo "all: $ALL" >> Makefile

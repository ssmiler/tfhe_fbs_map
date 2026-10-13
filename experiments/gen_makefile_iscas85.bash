#! /bin/bash

# wget benchmarks
[ -d benchmarks/iscas85/ ] || ( echo "Wget benchmarks" && wget -nd -r -l1 -A "*.bench" https://pld.ttu.ee/~maksim/benchmarks/iscas85/bench -P benchmarks/iscas85/ ) || exit

BENCHES=$(ls benchmarks/iscas85/*.bench)

FBS_SIZES=$(seq 2 32)

FBS_MAP="../target/release/fbs-map"

OUTPUT_DIR=outputs/iscas85

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
    echo -e "\t$FBS_MAP $BLIF --fbs-size $FBS_SIZE --mapper $MAPPER --output $OUT --output-lbf ${OUT_LBF} > $LOG 2>&1"
    echo ""
}

for BLIF in $BENCHES
do
    BENCH=$(basename -- "$BLIF" .bench)

    for MAPPER in "basic"
    do
        run_bench $BLIF $BENCH 2 $MAPPER >> Makefile
    done

    for FBS_SIZE in $FBS_SIZES
    do
        for MAPPER in "search"
        do
            run_bench $BLIF $BENCH $FBS_SIZE $MAPPER >> Makefile
        done
    done

done

echo ".DEFAULT_GOAL := all" >> Makefile
echo "all: $ALL" >> Makefile

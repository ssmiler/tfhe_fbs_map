#! /bin/bash

# clone EPFL benchmarks
[ -d benchmarks/bristol ] || ( echo "Clone bristol circuit benchmarks" && git clone https://github.com/mkskeller/bristol-fashion.git benchmarks/bristol ) || exit

BENCHES=""
# BENCHES+=" benchmarks/bristol/adder64.txt"
BENCHES+=" benchmarks/bristol/aes_128.txt"
# BENCHES+=" benchmarks/bristol/aes_192.txt"
# BENCHES+=" benchmarks/bristol/aes_256.txt"
BENCHES+=" benchmarks/bristol/AES-non-expanded.txt"
# BENCHES+=" benchmarks/bristol/divide64.txt"
# BENCHES+=" benchmarks/bristol/FP-add.txt"
# BENCHES+=" benchmarks/bristol/FP-div.txt"
# BENCHES+=" benchmarks/bristol/FP-eq.txt"
# BENCHES+=" benchmarks/bristol/FP-f2i.txt"
# BENCHES+=" benchmarks/bristol/FP-i2f.txt"
# BENCHES+=" benchmarks/bristol/FP-mul.txt"
# BENCHES+=" benchmarks/bristol/FP-sqrt.txt"
# BENCHES+=" benchmarks/bristol/ModAdd512.txt"
# BENCHES+=" benchmarks/bristol/mult2_64.txt"
# BENCHES+=" benchmarks/bristol/mult64.txt"
# BENCHES+=" benchmarks/bristol/neg64.txt"
# BENCHES+=" benchmarks/bristol/sub64.txt"
# BENCHES+=" benchmarks/bristol/udivide64.txt"
# BENCHES+=" benchmarks/bristol/zero_equal.txt"

# BENCHES+=" benchmarks/bristol/Keccak_f.txt"
# BENCHES+=" benchmarks/bristol/LSSS_to_GC.txt"
# BENCHES+=" benchmarks/bristol/sha256.txt"
# BENCHES+=" benchmarks/bristol/sha512.txt"

FBS_SIZES=$(seq 3 16)

MAP_CIRCUIT_PY="../fbs_mapper/map_circuit.py"

MAPPERS="naive search"

OUTPUT_DIR=outputs/bristol

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

for BENCH in $BENCHES
do
    BASE=$(basename -- "$BENCH" .txt)

    FBS_SIZE=2
    for MAPPER in "basic" "search"
    do
        run_bench $BLIF $BENCH $FBS_SIZE $MAPPER >> Makefile
    done

    for FBS_SIZE in $FBS_SIZES
    do
        for MAPPER in $MAPPERS
        do
            run_bench $BLIF $BENCH $FBS_SIZE $MAPPER >> Makefile
        done
    done

done

echo ".DEFAULT_GOAL := all" >> Makefile
echo "all: $ALL" >> Makefile

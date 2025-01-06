#! /bin/bash

# clone and compile abc
[ -f abc/abc ] || ( echo "Clone and compile abc" && git clone https://github.com/berkeley-abc/abc && cd abc && make -j4 abc ) || exit

# wget benchmarks
[ -d benchmarks/iscas89/ ] || ( echo "Wget benchmarks" && wget -nd -r -l1 -A "*.bench" https://pld.ttu.ee/~maksim/benchmarks/iscas89/bench -P benchmarks/iscas89/ ) || exit

BENCHES=""
# BENCHES+=" benchmarks/iscas89/c1196.bench"
# BENCHES+=" benchmarks/iscas89/c1238.bench"
# BENCHES+=" benchmarks/iscas89/s1196a.bench"
# BENCHES+=" benchmarks/iscas89/s1196b.bench"
BENCHES+=" benchmarks/iscas89/s1196.bench"
# BENCHES+=" benchmarks/iscas89/s1238a.bench"
BENCHES+=" benchmarks/iscas89/s1238.bench"
BENCHES+=" benchmarks/iscas89/s13207.bench"
BENCHES+=" benchmarks/iscas89/s1423.bench"
BENCHES+=" benchmarks/iscas89/s1488.bench"
BENCHES+=" benchmarks/iscas89/s1494.bench"
BENCHES+=" benchmarks/iscas89/s15850.bench"
# BENCHES+=" benchmarks/iscas89/s208a.bench"
BENCHES+=" benchmarks/iscas89/s208.bench"
# BENCHES+=" benchmarks/iscas89/s27a.bench"
BENCHES+=" benchmarks/iscas89/s27.bench"
BENCHES+=" benchmarks/iscas89/s298.bench"
BENCHES+=" benchmarks/iscas89/s344.bench"
BENCHES+=" benchmarks/iscas89/s349.bench"
BENCHES+=" benchmarks/iscas89/s35932.bench"
BENCHES+=" benchmarks/iscas89/s382.bench"
BENCHES+=" benchmarks/iscas89/s38417.bench"
BENCHES+=" benchmarks/iscas89/s38584.bench"
# BENCHES+=" benchmarks/iscas89/s386a.bench"
BENCHES+=" benchmarks/iscas89/s386.bench"
BENCHES+=" benchmarks/iscas89/s400.bench"
# BENCHES+=" benchmarks/iscas89/s420a.bench"
BENCHES+=" benchmarks/iscas89/s420.bench"
BENCHES+=" benchmarks/iscas89/s444.bench"
BENCHES+=" benchmarks/iscas89/s510.bench"
# BENCHES+=" benchmarks/iscas89/s526a.bench"
BENCHES+=" benchmarks/iscas89/s526.bench"
BENCHES+=" benchmarks/iscas89/s5378.bench"
BENCHES+=" benchmarks/iscas89/s641.bench"
BENCHES+=" benchmarks/iscas89/s713.bench"
BENCHES+=" benchmarks/iscas89/s820.bench"
BENCHES+=" benchmarks/iscas89/s832.bench"
# BENCHES+=" benchmarks/iscas89/s838a.bench"
BENCHES+=" benchmarks/iscas89/s838.bench"
BENCHES+=" benchmarks/iscas89/s9234.bench"
BENCHES+=" benchmarks/iscas89/s953.bench"

FBS_SIZES=$(seq 3 32)

MAP_CIRCUIT_PY="../fbs_mapper/map_circuit.py"

MAPPERS="naive search"

BENCH_XAG_DIR=outputs/benchmarks_xag/iscas89
OUTPUT_DIR=outputs/iscas89

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
    BENCH_BASE=$(basename -- "$BENCH" .bench)
    BLIF="$BENCH_XAG_DIR/$BENCH_BASE.blif"
    BLIF_XAG="$BENCH_XAG_DIR/${BENCH_BASE}-xag.blif"

    echo "$BLIF: $BENCH | $BENCH_XAG_DIR" >> Makefile
    echo -e "\t./abc/abc -c \"read_bench $BENCH; frames -F 10 -i; write_blif $BLIF\"" >> Makefile
    echo >> Makefile
    BLIFS+=" $BLIF"

    # echo "$BLIF_XAG: $BENCH | $BENCH_XAG_DIR" >> Makefile
    # echo -e "\t./abc/abc -c \"read_bench $BENCH; frames -F 10 -i; read_library lib.genlib; ps; map; ps; unmap; ps; write_blif $BLIF_XAG\"" >> Makefile
    # echo >> Makefile
    # BLIFS+=" $BLIF_XAG"
done
ALL+=" $BLIFS"

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
    OUT_LBF="$OUTPUT_DIR/$BASE"_"$FBS_SIZE"_"$MAPPER.lbf"
    LOG="$OUTPUT_DIR/$BENCH"_"$FBS_SIZE"_"$MAPPER.log"
    ALL+=" ${OUT_LBF}"

    echo "$OUT $OUT_LBF $LOG: $BLIF | $OUTPUT_DIR"
    echo -e "\tpython3 $MAP_CIRCUIT_PY $BLIF --fbs_size $FBS_SIZE --mapper $MAPPER --output $OUT --output_lbf ${OUT_LBF} > $LOG 2>&1"
    echo ""
}

for BLIF in $BLIFS
do
    BENCH=$(basename -- "$BLIF" .blif)

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

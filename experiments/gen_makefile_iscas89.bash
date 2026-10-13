#! /bin/bash

# wget benchmarks
[ -d benchmarks/iscas89 ] || ( echo "Wget benchmarks" && wget -nd -r -l1 -A "*.bench" https://pld.ttu.ee/~maksim/benchmarks/iscas89/bench -P benchmarks/iscas89 ) || exit

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

FBS_SIZES=$(seq 2 32)

FBS_MAP="../target/release/fbs-map"

BENCH_UNROLLED_DIR=outputs/iscas89_unrolled
OUTPUT_DIR=outputs/iscas89

rm -f Makefile
ALL=""

# show current Makefile
echo "log_makefile:" >> Makefile
echo -e "\t@cat Makefile" >> Makefile
echo -e "\t@echo" >> Makefile
echo -e "\t@echo" >> Makefile
echo >> Makefile

# targets for unrolling sequential benchmarks into combinational BENCH files
echo "$BENCH_UNROLLED_DIR:" >> Makefile
echo -e "\t@mkdir -p $BENCH_UNROLLED_DIR" >> Makefile
echo >> Makefile
ALL+=" $BENCH_UNROLLED_DIR"

UNROLLED_BENCHES=""
for BENCH in $BENCHES
do
    BENCH_BASE=$(basename -- "$BENCH" .bench)
    UNROLLED="$BENCH_UNROLLED_DIR/$BENCH_BASE.bench"

    echo "$UNROLLED: $BENCH unroll_bench.py | $BENCH_UNROLLED_DIR" >> Makefile
    echo -e "\tpython3 unroll_bench.py $BENCH $UNROLLED --frames 10" >> Makefile
    echo >> Makefile
    UNROLLED_BENCHES+=" $UNROLLED"
done
ALL+=" $UNROLLED_BENCHES"

# targets for mapping bench circuits to FBSs
echo "$OUTPUT_DIR:" >> Makefile
echo -e "\t@mkdir -p $OUTPUT_DIR" >> Makefile
echo >> Makefile
ALL+=" $OUTPUT_DIR"

function run_bench() {
    local CIRCUIT=$1
    local BENCH_NAME=$2
    local FBS_SIZE=$3
    local MAPPER=$4

    OUT="$OUTPUT_DIR/$BENCH_NAME"_"$FBS_SIZE"_"$MAPPER.fbs"
    OUT_LBF="$OUTPUT_DIR/$BENCH_NAME"_"$FBS_SIZE"_"$MAPPER.lbf"
    LOG="$OUTPUT_DIR/$BENCH_NAME"_"$FBS_SIZE"_"$MAPPER.log"
    ALL+=" ${OUT_LBF}"

    echo "$OUT $OUT_LBF $LOG: $CIRCUIT | $OUTPUT_DIR"
    echo -e "\t$FBS_MAP $CIRCUIT --fbs-size $FBS_SIZE --mapper $MAPPER --output $OUT --output-lbf ${OUT_LBF} > $LOG 2>&1"
    echo ""
}

for CIRCUIT in $UNROLLED_BENCHES
do
    BENCH=$(basename -- "$CIRCUIT" .bench)

    for MAPPER in "basic"
    do
        run_bench $CIRCUIT $BENCH 2 $MAPPER >> Makefile
    done

    for FBS_SIZE in $FBS_SIZES
    do
        for MAPPER in "search"
        do
            run_bench $CIRCUIT $BENCH $FBS_SIZE $MAPPER >> Makefile
        done
    done

done

echo ".DEFAULT_GOAL := all" >> Makefile
echo "all: $ALL" >> Makefile

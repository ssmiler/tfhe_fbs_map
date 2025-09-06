NB_J=4

OUT_DIR=outputs

mkdir -p ${OUT_DIR}
rm -f ${OUT_DIR}/*.log

echo "Executing heuristic on EPFL benchmarks"
bash gen_makefile_epfl.bash
mv Makefile ${OUT_DIR}/makefile_epfl
make -f ${OUT_DIR}/makefile_epfl -j $NB_J >> ${OUT_DIR}/epfl.log 2>&1
python3 build_csv.py -o ${OUT_DIR}/epfl_agg.csv ${OUT_DIR}/epfl/

echo "Executing heuristic on iscas85 benchmarks"
bash gen_makefile_iscas85.bash
mv Makefile ${OUT_DIR}/makefile_iscas85
make -f ${OUT_DIR}/makefile_iscas85 -j $NB_J >> ${OUT_DIR}/iscas85.log 2>&1
python3 build_csv.py -o ${OUT_DIR}/iscas85_agg.csv ${OUT_DIR}/iscas85/

echo "Executing heuristic on iscas89 benchmarks"
bash gen_makefile_iscas89.bash
mv Makefile ${OUT_DIR}/makefile_iscas89
make -f ${OUT_DIR}/makefile_iscas89 -j $NB_J >> ${OUT_DIR}/iscas89.log 2>&1
python3 build_csv.py -o ${OUT_DIR}/iscas89_agg.csv ${OUT_DIR}/iscas89/

echo "Executing heuristic on generated benchmarks (trivium, kreyvium, aes-sbox, etc.)"
bash gen_makefile_generated.bash
mv Makefile ${OUT_DIR}/makefile_generated
make -f ${OUT_DIR}/makefile_generated -j $NB_J >> ${OUT_DIR}/generated.log 2>&1
python3 build_csv.py -o ${OUT_DIR}/generated_agg.csv ${OUT_DIR}/generated/

echo "Executing heuristic on some bristol benchmarks"
bash gen_makefile_bristol.bash
mv Makefile ${OUT_DIR}/makefile_bristol
make -f ${OUT_DIR}/makefile_bristol -j $NB_J >> ${OUT_DIR}/bristol.log 2>&1
python3 build_csv.py -o ${OUT_DIR}/bristol_agg.csv ${OUT_DIR}/bristol/

# add execution estimates
echo "Compute and add execution time estimates to mapped circuits"
python3 add_exec_estimates.py ${OUT_DIR}/*_agg.csv

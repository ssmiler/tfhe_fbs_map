# Heuristic for mapping Boolean circuit to TFHE functional bootstrappings

This repository contains the code used in paper *[A fast heuristic for mapping Boolean circuits to functional bootstrapping](https://eprint.iacr.org/2024/1204)*.

## How to run the experiments

### Preliminaries

Go to `experiments` directory:
```bash
cd experiments
```

Clone and patch concrete to support non power-of-two parameters:
```bash
git clone https://github.com/zama-ai/concrete.git -b nightly-2024.04.17
(cd concrete; git apply ../concrete.patch)
```

Install required python packages:
```bash
pip3 install -r requirements.txt
```

> [!NOTE]
> Minimum Python version `3.10` required 

### Run benchmarks

```bash
bash run_benchmarks.sh
```

### Analyse results

```bash
cd outputs
python3 analyse_results.py
```

import sys
import random
import argparse
import time
import logging
import traceback

from bit_exec_env import *
import map_to_fbs


def get_truth_table(blif_parser_tt):
    ttl = blif_parser_tt[0]
    n = len(ttl) - 1
    assert(len(ttl) > 0)
    tte = 0 if ttl[-1] == '1' else 1
    truth_table = [tte] * (2 ** n)
    for l in blif_parser_tt:
        assert(len(ttl) == len(l))
        k = sum(map(lambda k: int(l[k]) * 2 ** (n - k - 1), range(n)))
        truth_table[k] = 1 - tte
    return truth_table


def parse_blif(filename: str):
    import blifparser.blifparser as blifparser

    blif = blifparser.BlifParser(filename).blif

    bit_env = BitExecEnv()

    wires = dict(
        map(lambda name: (name, bit_env.input(name)), blif.inputs.inputs))

    for gate in blif.booleanfunctions:
        inps = list(map(lambda k: wires[k], gate.inputs))
        truth_table = get_truth_table(gate.truthtable)
        if truth_table == [0]:
            wires[gate.output] = bit_env.CONST0
        elif truth_table == [1]:
            wires[gate.output] = bit_env.CONST1
        else:
            assert(len(truth_table) == 4 or len(truth_table) == 2), gate
            wires[gate.output] = bit_env.op_lut(
                inps, truth_table, name=gate.output)

    for out in blif.outputs.outputs:
        bit_env.output(out, wires[out])

    return bit_env


def parse_bristol(filename: str):
    # https://nigelsmart.github.io/MPC-Circuits/

    import bfcl

    c = bfcl.circuit()
    c.parse(open(filename).read())

    bit_env = BitExecEnv()

    wires = dict(
        map(lambda idx: (idx, bit_env.input(f"i_{idx}")), c.wire_in_index))

    for gate in c.gate:
        assert(len(gate.wire_out_index) == 1)
        assert(len(gate.wire_in_index) == 1 or len(gate.wire_in_index) == 2)
        out_idx = gate.wire_out_index[0]
        if gate.operation:
            inps = list(map(lambda k: wires[k], gate.wire_in_index))
            truth_table = list(gate.operation)
            if truth_table == [0]:
                wires[gate.output] = bit_env.CONST0
            elif truth_table == [1]:
                wires[gate.output] = bit_env.CONST1
            else:
                assert(len(truth_table) == 4 or len(truth_table) == 2), gate
                wires[out_idx] = bit_env.op_lut(
                    inps, truth_table, name=f"w_{out_idx}")
        else:
            assert(len(gate.wire_in_index) == 1)
            in_idx = gate.wire_in_index[0]
            wires[out_idx] = wires[in_idx]

    for out_idx in c.wire_out_index:
        bit_env.output(out_idx, wires[out_idx])

    return bit_env


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Map logic gates to Functional Boostrapping (FBS)")
    parser.add_argument("filename", help="input circuit")
    parser.add_argument(
        "--type", choices=["blif", "bristol"], default="blif", help="format")
    parser.add_argument("--fbs_size", default=3,
                        type=int, help="FBS size")
    parser.add_argument(
        "--mapper", choices=["basic", "naive", "search"], default="search", help="mapping strategy")
    parser.add_argument("--strict_fbs_size", action="store_true",
                        help="do not use anti-cyclic ring property")
    parser.add_argument("--output", help="output mapped circuit file")
    parser.add_argument("--output_lbf", help="output mapped circuit file in LBS format")
    parser.add_argument("--max_tt_size", default=16, type=int,
                        help="maximal truth table size (log2) before bootstrapping")
    parser.add_argument("--verbose", "-v", action="count", default=0)

    args = parser.parse_args()

    levels = [logging.CRITICAL, logging.ERROR,
              logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(args.verbose, len(levels) - 1)]
    logging.basicConfig(level=level)

    fbs_size = args.fbs_size
    if args.strict_fbs_size:
        max_fbs_size = fbs_size
    else:
        max_fbs_size = 2 * fbs_size
    args.max_fbs_size = max_fbs_size

    match args.mapper:
        case "basic": mapper = map_to_fbs.MapToFBSBasic()
        case cone_merger:
            mapper = map_to_fbs.MapToFBSHeur(
                fbs_size=fbs_size,
                max_fbs_size=max_fbs_size,
                max_truth_table_size=args.max_tt_size,
                cone_merger=cone_merger)

    match args.type:
        case "blif": bit_env = parse_blif(args.filename)
        case "bristol": bit_env = parse_bristol(args.filename)

    np.random.seed(42)
    input_vals = dict(
        map(lambda inp: (inp.name, np.random.randint(0, 2, (1000))), bit_env.inputs))
    output_values1 = bit_env.eval(input_vals)

    # bit_env.print()
    # print(bit_env.stats())

    start = time.time()
    try:
        lut_env = mapper.map(bit_env)
    except Exception as e:
        logging.critical(traceback.format_exc())
        sys.exit()

    lut_env.remove_dangling_nodes()
    duration = time.time() - start

    stats = lut_env.stats()
    stats.update(args.__dict__)
    stats["time"] = duration

    print(stats)

    # print(f"test1 = {output_values1['test']}")
    # print(f"test2 = {output_values2['test']}")
    # assert(np.all(output_values1["test"] == output_values2["test"]))

    # print(f"Input values:")
    # print('\n\t'.join(map(lambda e: f'{e[0]} : {e[1]}', input_vals.items())))

    # print(f"Output values 1:")
    # print('\n\t'.join(map(lambda e: f'{e[0]} : {e[1]}', output_values1.items())))

    # print(f"Output values 2:")
    # print('\n\t'.join(map(lambda e: f'{e[0]} : {e[1]}', output_values2.items())))

    output_values2 = lut_env.eval(input_vals)
    assert(output_values1.keys() == output_values2.keys())
    for k in output_values1.keys():
        equal = np.all(output_values1[k] == output_values2[k])
        if not equal:
            print(f"output {k} do not match {output_values1[k]} {output_values2[k]}")
        assert(equal)

    if args.output is not None:
        with open(args.output, "w") as file:
            lut_env.print(show_outputs=True, os=file)

    if args.output_lbf is not None:
        with open(args.output_lbf, "w") as file:
            lut_env.write_lbf(os=file)

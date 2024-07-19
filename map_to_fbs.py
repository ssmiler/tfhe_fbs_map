from bit_exec_env import *
from lut_exec_env import *
import numpy as np
import itertools as it
import logging

############################
# TODOs:
#   - when mvt is sparse (i.e. mvt size is larger than unique mvt values) try to
#     fit "don't care" (i.e. tt values for which non mvt values exists) for
#     use_anticyclic_ring
#


class MapToLUTBasic:
    # map each gate into a linear combination + boostrapping
    def map(self, env: BitExecEnv):
        lut_env = LutExecEnv()

        wires = {"0": lut_env.const(0), "1": lut_env.const(1)}
        for instr in env.instructions:
            logging.getLogger("MapToLUTBasic").info(f"{instr}")
            match instr:
                case BitExecEnv.Const(val=val):
                    wires[name] = lut_env.const(val)
                case BitExecEnv.Input(name=name):
                    wires[name] = lut_env.input(name)
                case BitExecEnv.LUT(name=name, inputs=inputs, truth_table=table):
                    assert(len(table) == 2 ** len(inputs))

                    if len(inputs) == 1:
                        wire = wires[inputs[0].name]
                        if np.all(table == [1, 0]):
                            wires[name] = lut_env.linear(
                                [-1], [wire], const_coef=1)
                        else:
                            assert(np.all(table == [0, 1]))
                            wires[name] = wire
                    else:
                        coefs = list(
                            map(lambda k: 2**k, range(len(inputs))))[::-1]
                        vals = list(map(lambda e: wires[e.name], inputs))
                        l = lut_env.linear(coefs, vals)
                        wires[name] = lut_env.bootstrap(l, table)
                case _:
                    assert(False), "Unknown instruction"

        for name, out in env.outputs.items():
            lut_env.output(name, wires[out.name])

        return lut_env


class MapToLUTBase:
    def __init__(self, fbs_size=8, max_fbs_size = 16):
        self.fbs_size = fbs_size
        self.max_fbs_size = max_fbs_size

    def _mvt_size(self, mvt):
        return max(mvt) - min(mvt) + 1

    def _comp_boot_test_vector(self, tt, mvt):
        tmp = dict(it.product(range(self._mvt_size(mvt)), [0]))
        tmp.update(zip(mvt, tt))
        return list(map(lambda e: e[1], sorted(tmp.items())))

    def _is_mvt_valid(self, tt, mvt):
        return len(set(mvt[tt == 0]).intersection(mvt[tt == 1])) == 0

    def _is_lut_valid(self, tt, mvt):
        if not self._is_mvt_valid(tt, mvt):
            return False
        tt_lut = self._comp_boot_test_vector(tt, mvt)
        if len(tt_lut) <= self.fbs_size:
            return True
        if len(tt_lut) <= self.max_fbs_size:
            tt_lut = np.array(tt_lut)
            tt_start = tt_lut[0:len(tt_lut) - self.fbs_size]
            tt_end = tt_lut[self.fbs_size:]

            # f(x) == -f(x + fbs_size)
            mode1 = np.all(tt_start != tt_end)
            # f(x) == 0 == -f(x + fbs_size)
            mode2 = np.all(tt_start == tt_end) and np.all(0 == tt_start)
            # f(x) == 1 == -f(x + fbs_size)
            mode3 = np.all(tt_start == tt_end) and np.all(1 == tt_start)
            return mode1 or mode2 or mode3
        return False

    def new_lut_exec_env(self):
        return LutExecEnv(merge_linear_prods=False)

    # overriden in child class
    def new_wire(self, *args):
        pass

    # overriden in child class
    def new_const(self, lut_env, cst):
        pass

    # overriden in child class
    def new_bootstrap(self, lut_env, inp):
        pass

    # overriden in child class
    def treat_bit_exec_lut_gate(self, lut_env, input_wires, truth_table):
        pass

    def map(self, env: BitExecEnv):
        pass

    def compute_truth_table(self, tt, truth_table):
        return np.fromiter(map(lambda e: truth_table[e], tt), dtype=np.uint8)

    def compute_truth_table_cat(self, tt1, tt2, truth_table):
        return np.fromiter(map(lambda e: truth_table[2*e[0] + e[1]], zip(tt1, tt2)), dtype=np.uint8)

    def compute_truth_table_prod(self, tt1, tt2, truth_table):
        return np.fromiter(map(lambda e: truth_table[2*e[0] + e[1]], it.product(tt1, tt2)), dtype=np.uint8)

    def map_internal(self, env: BitExecEnv, nodes_to_bootstrap):
        logger = self.logger.getChild("map_internal")

        lut_env = self.new_lut_exec_env()

        wires = {"0": self.new_const(
            lut_env, 0), "1": self.new_const(lut_env, 1)}
        for instr in env.instructions:
            match instr:
                case BitExecEnv.Const(name=name):
                    wire = wires[name]
                case BitExecEnv.Input(name=name):
                    wire = self.new_wire(lut_env.input(name))
                case BitExecEnv.LUT(name=name, inputs=inputs, truth_table=truth_table):
                    assert(len(inputs) <=
                           2), "only 1 or 2 input gates are supported"
                    input_wires = list(
                        map(lambda inp: wires[inp.name], inputs))

                    if len(input_wires) == 1:
                        logger.info(f"({name}) <- {input_wires[0].name()} ({inputs[0].name})")
                    else:
                        logger.info(f"({name}) <- {input_wires[0].name()} ({inputs[0].name}) {input_wires[1].name()} ({inputs[1].name})")

                    wire, bw = self.treat_bit_exec_lut_gate(
                        lut_env, input_wires, truth_table)
                    logger.info(f"\t{wire.name()} {wire.val} {wire.tt}")

                    for inp_idx, inp_new_wire in bw.items():
                        be_name = inputs[inp_idx].name
                        logger.info(f"\t\t{be_name} {inp_new_wire.name()}")
                        wires[be_name] = inp_new_wire

                case _:
                    assert(False), "Unknown instruction"

            name = instr.name
            if name in nodes_to_bootstrap:
                wires[name] = self.new_bootstrap(lut_env, wire)
            else:
                wires[name] = wire

        for name, out in env.outputs.items():
            wire = self.new_bootstrap(lut_env, wires[out.name])
            lut_env.output(name, wire.val)

        return lut_env

    def map(self, env: BitExecEnv):
        nodes_to_bootstrap = set(map(lambda e: e.name, env.outputs.values()))
        lut_env = self.map_internal(env, nodes_to_bootstrap)
        return lut_env


class MapToLUTNaive(MapToLUTBase):
    # map each 2-input gate to FB with powers-of-2 coefficients

    def __init__(self, fbs_size=8, max_fbs_size=16):
        super().__init__(fbs_size=fbs_size, max_fbs_size=max_fbs_size)
        self.logger = logging.getLogger("MapToLUTNaive")

    def new_wire(self, val, tt=[0, 1]):
        class Wire:
            def __init__(self, val, tt):
                self.val = val
                self.tt = tt

            def size(self):
                return len(self.tt)

            def name(self):
                return self.val.name

            def __eq__(self, other):
                return self.name() == other.name() and np.all(self.tt == other.tt)

        return Wire(val, tt)

    def new_const(self, lut_env, cst):
        return self.new_wire(lut_env.const(cst), [cst])

    def _negate(self, lut_env, inp):
        return self.new_wire(lut_env.linear([-1], [inp.val], const_coef=1), [0, 1])

    def new_bootstrap(self, lut_env, inp):
        if inp.size() == 1:
            return inp
        elif inp.size() == 2:
            if np.all(inp.tt == [1, 0]):
                return self._negate(lut_env, inp)
            else:
                return inp
        elif np.all(inp.tt == 0):
            return self.new_const(lut_env, 0)
        elif np.all(inp.tt == 1):
            return self.new_const(lut_env, 1)

        return self.new_wire(lut_env.bootstrap(inp.val, inp.tt.tolist()))

    def _add_linear(self, lut_env, inp1, inp2, tt):
        assert(len(tt) > 1)
        if inp1.size() == 1:  # inp1 is constant, pass-through
            return self.new_wire(inp2.val, tt)
        elif inp2.size() == 1:  # inp2 is constant, pass-through
            return self.new_wire(inp1.val, tt)

        a, b = inp2.size(), 1
        return self.new_wire(lut_env.linear([a, b], [inp1.val, inp2.val]), tt)

    def treat_bit_exec_lut_gate(self, lut_env, input_wires, truth_table):
        logger = self.logger.getChild("treat_bit_exec_lut_gate")

        if len(input_wires) == 1:
            inp, = input_wires
            logger.info(f"1-input gate:")
            logger.info(f"\tgate tt: {truth_table}")
            logger.info(f"\t{inp.name()} ({inp.size()})")
            logger.info(f"\t\t{inp.tt}")
            assert(len(truth_table) == 2), "error"
            r_tt = self.compute_truth_table(inp.tt, truth_table)
            logger.info(f"\toutput tt {r_tt}")
            return self.new_wire(inp.val, self.compute_truth_table(inp.tt, truth_table)), {}

        assert(len(input_wires) == 2), f"error {input_wires}"
        assert(len(truth_table) == 4), f"error {truth_table}"

        inp1, inp2 = input_wires

        logger.info(f"2-input gate:")
        logger.info(f"\tgate tt: {truth_table}")
        logger.info(f"\t{inp1.name()} ({inp1.size()})")
        logger.info(f"\t\t{inp1.tt}")
        logger.info(f"\t{inp2.name()} ({inp2.size()})")
        logger.info(f"\t\t{inp2.tt}")

        # same input twice, pass-through
        if inp1 == inp2:
            r_tt = self.compute_truth_table_cat(inp1.tt, inp2.tt, truth_table)
            logger.info(f"same-input gate:")
            logger.info(f"\t{r_tt}")
            if np.all(r_tt == r_tt[0]):
                return self.new_const(lut_env, r_tt[0]), {}
            return self.new_wire(inp1.val, r_tt), {}

        # if needed bootstrap input wires, start with largest truth table
        if inp1.size() < inp2.size():
            inp1, inp2 = inp2, inp1
            truth_table[1], truth_table[2] = truth_table[2], truth_table[1]
            logger.info(f"switch inp1 and inp2:")
            logger.info(f"\tgate tt: {truth_table}")

        r_tt = self.compute_truth_table_prod(inp1.tt, inp2.tt, truth_table)
        r_mvt = np.array(list(range(inp1.size() * inp2.size())))
        logger.info(f"output tt and mvt:")
        logger.info(f"\t{r_tt}")
        logger.info(f"\t{r_mvt}")

        if np.all(r_tt == r_tt[0]):
            return self.new_const(lut_env, r_tt[0]), {}
        if not self._is_lut_valid(r_tt, r_mvt):
            logger.info(f"bootstrap {inp1.name()}")
            inp1 = self.new_bootstrap(lut_env, inp1)
            r_tt = self.compute_truth_table_prod(inp1.tt, inp2.tt, truth_table)
            r_mvt = np.array(list(range(inp1.size() * inp2.size())))

            logger.info(f"\tnew wire {inp1.name()} {inp1.tt}")
            logger.info(f"\t{r_tt}")

            if np.all(r_tt == r_tt[0]):
                return self.new_const(lut_env, r_tt[0]), {}
            if not self._is_lut_valid(r_tt, r_mvt):
                logger.info(f"bootstrap {inp2.name()}")
                inp2 = self.new_bootstrap(lut_env, inp2)
                r_tt = self.compute_truth_table_prod(
                    inp1.tt, inp2.tt, truth_table)

                logger.info(f"\tnew wire {inp2.name()} {inp2.tt}")
                logger.info(f"\t{r_tt}")
                if np.all(r_tt == r_tt[0]):
                    return self.new_const(lut_env, r_tt[0]), {}

        logger.info(f"final:")
        logger.info(f"\tgate tt: {truth_table}")
        logger.info(f"\t{inp1.name()} {inp1.tt} ({len(inp1.tt)} {inp1.size()})")
        logger.info(f"\t{inp2.name()} {inp2.tt} ({len(inp2.tt)} {inp2.size()})")
        logger.info(f"\toutput tt {r_tt} and mvt {r_mvt}")

        return self._add_linear(lut_env, inp1, inp2, r_tt), {}


class MapToLUTSearch(MapToLUTBase):
    # map each 2-input gate to FB with minimal coefficients

    def __init__(self, fbs_size=8, max_fbs_size=16, max_truth_table_size=2**16):
        super().__init__(fbs_size=fbs_size, max_fbs_size=max_fbs_size)
        self._find_optim_truth_table_cache = dict()
        self.max_truth_table_size = max_truth_table_size
        self.logger = logging.getLogger("MapToLUTSearch")

    def lincomb_norm2(self, a, inp1, b, inp2):
        return a*a*inp1.norm2 + b*b*inp2.norm2

    def new_wire(self, *args):
        class Wire:
            def __init__(self, parent, val, tt=[0, 1], mvt=[0, 1], norm2=1):
                self.parent = parent
                self.val = val
                self.tt = np.array(tt)
                self.mvt = np.array(mvt)
                self.norm2 = norm2

            def size(self):
                return self.parent._mvt_size(self.mvt)

            def name(self):
                return self.val.name

            def __eq__(self, other):
                return self.name() == other.name() and np.all(self.tt == other.tt)

        wire = Wire(self, *args)
        # print(f"{wire.name()} {wire.norm2}")
        return wire

    def new_const(self, lut_env, cst):
        return self.new_wire(lut_env.const(cst), [cst], [0], 0)

    def _negate(self, lut_env, inp):
        return self.new_wire(lut_env.linear([-1], [inp.val], const_coef=1), [0, 1], inp.mvt, inp.norm2)

    def new_bootstrap(self, lut_env, inp):
        # assert(self._is_lut_valid(inp.tt, inp.mvt))
        if inp.size() == 1:
            return inp
        elif inp.size() == 2:
            if np.all(inp.tt == [1, 0]):
                return self._negate(lut_env, inp)
            else:
                return inp
        elif np.all(inp.tt == 0):
            return self.new_wire(lut_env.const(0), [0], [0], 0)
        elif np.all(inp.tt == 1):
            return self.new_wire(lut_env.const(1), [1], [1], 0)

        lut_tt = self._comp_boot_test_vector(inp.tt, inp.mvt)
        new_val = lut_env.bootstrap(inp.val, lut_tt)
        return self.new_wire(new_val)

    def _find_optim_truth_table_search(self, inp1, inp2, r_tt):
        r_abc = None
        r_mvt = None
        r_mvt_max = 1000000000
        r_norm2 = 1000000000

        xy_mvt = np.array(list(it.product(inp1.mvt, inp2.mvt)))

        inp1_size_m1, inp2_size_m1 = inp1.size() - 1, inp2.size() - 1
        to_try = set([(inp2_size_m1 + 1, inp1_size_m1 + 1)])
        tried = set()
        while to_try:
            a, b = to_try.pop()
            tried.add((a, b))

            ab = None
            if a > 1:
                ab = (a-1, b)
            elif a == 1:
                ab = (-1, b)
            elif a >= -inp2_size_m1:
                ab = (a-1, b)
            if ab and ab not in tried:
                to_try.add(ab)

            ab = None
            if b > 1:
                ab = (a, b-1)
            if ab and ab not in tried:
                to_try.add(ab)

            mvt_max = abs(a) * inp1_size_m1 + abs(b) * inp2_size_m1
            norm2 = self.lincomb_norm2(a, inp1, b, inp2)
            if mvt_max < r_mvt_max and norm2 < r_norm2:
                mvt = a * xy_mvt[:, 0] + b * xy_mvt[:, 1]
                c = -mvt.min()
                mvt += c
                if self._is_lut_valid(r_tt, mvt):
                    r_abc = (a, b, c)
                    r_mvt = mvt
                    r_mvt_max = mvt_max
                    r_norm2 = norm2

        return r_abc, r_mvt

    def _find_optim_truth_table(self, inp1, inp2, r_tt):
        key1a = f"{','.join(map(str,inp1.tt))}"
        key1b = f"{','.join(map(str,inp1.mvt))}"
        key2a = f"{','.join(map(str,inp2.tt))}"
        key2b = f"{','.join(map(str,inp2.mvt))}"
        key3 = f"{','.join(map(str,r_tt))}"
        key = f"{key1a}|{key1b}|{key2a}|{key2b}|{key3}"
        if key not in self._find_optim_truth_table_cache:
            self._find_optim_truth_table_cache[key] = self._find_optim_truth_table_search(
                inp1, inp2, r_tt)
        return self._find_optim_truth_table_cache[key]

    def _add_linear(self, lut_env, abc, inp1, inp2, tt, mvt):
        # assert(self._is_lut_valid(tt, mvt)
        #        ), f"BitExecEnv.LUT input truth table is not valid {abc} {inp1.name()} {inp2.name()} {tt} {mvt}"

        assert(len(tt) > 1)
        if inp1.size() == 1:  # inp1 is constant, pass-through
            return self.new_wire(inp2.val, tt, mvt, inp2.norm2)
        elif inp2.size() == 1:  # inp2 is constant, pass-through
            return self.new_wire(inp1.val, tt, mvt, inp1.norm2)

        a, b, c = abc
        new_val = lut_env.linear([a, b], [inp1.val, inp2.val], const_coef=c)
        return self.new_wire(new_val, tt, mvt, self.lincomb_norm2(a, inp1, b, inp2))

    def _switch_gate_inputs(self, inp1, inp2, idx1, idx2, truth_table):
        truth_table[1], truth_table[2] = truth_table[2], truth_table[1]
        return inp2, inp1, idx2, idx1, truth_table

    def treat_bit_exec_lut_gate(self, lut_env, input_wires, truth_table):
        logger = self.logger.getChild("treat_bit_exec_lut_gate")

        if len(input_wires) == 1:
            inp, = input_wires
            logger.info(f"1-input gate:")
            logger.info(f"\tgate tt: {truth_table}")
            logger.info(f"\t{inp.name()} ({len(inp.tt)} {inp.size()})")
            logger.info(f"\t\t{inp.tt}")
            logger.info(f"\t\t{inp.mvt}")
            assert(len(truth_table) == 2), "error"
            r_tt = self.compute_truth_table(inp.tt, truth_table)
            logger.info(f"\toutput tt {r_tt}")
            return self.new_wire(inp.val, r_tt, inp.mvt, inp.norm2), {}

        assert(len(input_wires) == 2), "error"
        assert(len(truth_table) == 4), "error"

        inp1, inp2 = input_wires

        logger.info(f"2-input gate:")
        logger.info(f"\tgate tt: {truth_table}")
        logger.info(f"\t{inp1.name()} ({len(inp1.tt)} {inp1.size()})")
        logger.info(f"\t\t{inp1.tt}")
        logger.info(f"\t\t{inp1.mvt}")
        logger.info(f"\t{inp2.name()} ({len(inp2.tt)} {inp2.size()})")
        logger.info(f"\t\t{inp2.tt}")
        logger.info(f"\t\t{inp2.mvt}")

        # same input twice, pass-through
        if inp1 == inp2:
            r_tt = self.compute_truth_table_cat(inp1.tt, inp2.tt, truth_table)
            logger.info(f"same-input gate:")
            logger.info(f"\t{r_tt}")
            if np.all(r_tt == r_tt[0]):
                return self.new_const(lut_env, r_tt[0]), {}
            return self.new_wire(inp1.val, r_tt, inp1.mvt, inp1.norm2), {}

        idx1, idx2 = 0, 1
        if inp1.size() < inp2.size():
            inp1, inp2, idx1, idx2, truth_table = self._switch_gate_inputs(
                inp1, inp2, idx1, idx2, truth_table)
            logger.info(f"switch inp1 and inp2:")
            logger.info(f"\tgate tt: {truth_table}")

        bootstrapped_wires = dict()

        # force bootstrap if output truth table is too large
        if len(inp1.tt) * len(inp2.tt) > self.max_truth_table_size:
            logger.info(f"bootstrap {inp1.name()} {idx1}")
            bootstrapped_wires[idx1] = inp1 = self.new_bootstrap(lut_env, inp1)
            inp1, inp2, idx1, idx2, truth_table = self._switch_gate_inputs(
                inp1, inp2, idx1, idx2, truth_table)

        if len(inp1.tt) * len(inp2.tt) > self.max_truth_table_size:
            logger.info(f"bootstrap {inp1.name()} {idx1}")
            bootstrapped_wires[idx1] = inp1 = self.new_bootstrap(lut_env, inp1)

        r_tt = self.compute_truth_table_prod(inp1.tt, inp2.tt, truth_table)
        if np.all(r_tt == r_tt[0]):
            return self.new_const(lut_env, r_tt[0]), bootstrapped_wires
        abc, r_mvt = self._find_optim_truth_table(
            inp1, inp2, r_tt)

        logger.info(f"optimal abc 1st:")
        logger.info(f"\t{abc}")
        logger.info(f"\t{r_tt}")
        logger.info(f"\t{r_mvt}")

        if abc == None:
            logger.info(f"bootstrap {inp1.name()} ({idx1})")
            bootstrapped_wires[idx1] = inp1 = self.new_bootstrap(lut_env, inp1)
        else:
            return self._add_linear(lut_env, abc, inp1, inp2, r_tt, r_mvt), bootstrapped_wires

        r_tt = self.compute_truth_table_prod(inp1.tt, inp2.tt, truth_table)
        if np.all(r_tt == r_tt[0]):
            return self.new_const(lut_env, r_tt[0]), bootstrapped_wires
        abc, r_mvt = self._find_optim_truth_table(
            inp1, inp2, r_tt)

        logger.info(f"optimal abc 2nd:")
        logger.info(f"\t{abc}")
        logger.info(f"\t{r_tt}")
        logger.info(f"\t{r_mvt}")

        if abc == None:
            logger.info(f"bootstrap {inp2.name()} ({idx2})")
            bootstrapped_wires[idx2] = inp2 = self.new_bootstrap(lut_env, inp2)
        else:
            return self._add_linear(lut_env, abc, inp1, inp2, r_tt, r_mvt), bootstrapped_wires

        r_tt = self.compute_truth_table_prod(inp1.tt, inp2.tt, truth_table)
        if np.all(r_tt == r_tt[0]):
            return self.new_const(lut_env, r_tt[0]), bootstrapped_wires
        abc, r_mvt = self._find_optim_truth_table(
            inp1, inp2, r_tt)

        logger.info(f"optimal abc 3rd:")
        logger.info(f"\t{abc}")
        logger.info(f"\t{r_tt}")
        logger.info(f"\t{r_mvt}")

        assert(abc)

        return self._add_linear(lut_env, abc, inp1, inp2, r_tt, r_mvt), bootstrapped_wires


if __name__ == '__main__':
    import itertools as it

    env = BitExecEnv()

    a = env.input("a")
    b = env.input("b")
    c = env.input("c")

    d = env.op_and(a, b)
    e = env.op_xor(c, d)
    f = env.op_lut([e, d], [0, 1, 0, 0])

    env.output("d", d)
    env.output("e", e)
    env.output("f", f)

    print("Input program:")
    env.print()

    input_vals = {"a": [0, 0, 1, 1], "b": [0, 1, 0, 1], "c": [0, 0, 1, 1]}
    print(env.eval(input_vals))

    print()

    print("Basic map:")
    lut_env = MapToLUTBasic().map(env)
    lut_env.print()

    print(lut_env.eval(input_vals))
    print()

    print("Naive map:")
    lut_env = MapToLUTNaive(fbs_size=8).map(env)
    lut_env.print()

    print(lut_env.eval(input_vals))
    print()

    print("Map using exhaustive search:")
    lut_env = MapToLUTSearch(fbs_size=8).map(env)
    lut_env.print()

    print(lut_env.eval(input_vals))
    print()

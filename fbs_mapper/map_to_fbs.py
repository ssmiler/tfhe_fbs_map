from bit_exec_env import *
from fbs_exec_env import *
import numpy as np
import itertools as it
import logging

############################
# TODOs:
#   - when mvt is sparse (i.e. mvt size is larger than unique mvt values) try to
#     fit "don't care" (i.e. tt values for which non mvt values exists) for
#     use_anticyclic_ring
#


class MapToFBSBasic:
    # map each gate into a linear combination + boostrapping
    def map(self, env: BitExecEnv):
        lut_env = LutExecEnv()

        wires = {"0": lut_env.const(0), "1": lut_env.const(1)}
        for instr in env.instructions:
            logging.getLogger("MapToFBSBasic").info(f"{instr}")
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


class MapToFBSHeur:
    # map each 2-input gate to FB with minimal coefficients

    def __init__(self, cone_merger, fbs_size=8, max_fbs_size=16, max_truth_table_size=16):
        self.fbs_size = fbs_size
        self.max_fbs_size = max_fbs_size
        self.max_truth_table_size = max_truth_table_size

        match cone_merger:
            case "naive": self._find_lincomb_coefs = self._find_lincomb_coefs_naive
            case "search": self._find_lincomb_coefs = self._find_lincomb_coefs_search
            case _: assert(False), f"Unknown cone merger '{cone_merger}'"

        self._find_lincomb_coefs_cache = dict()
        self.logger = logging.getLogger(f"MapToFBS_{cone_merger}")

    def _mvt_size(self, mvt):
        return np.max(mvt) - np.min(mvt) + 1

    def _comp_boot_test_vector(self, tt, mvt, missing_val):
        tmp = dict(it.product(range(mvt.min(), mvt.max() + 1), [missing_val]))
        tmp.update(zip(mvt, tt))
        return list(map(lambda e: e[1], sorted(tmp.items())))

    def _is_mvt_valid(self, tt, mvt):
        return len(set(mvt[tt == 0]).intersection(mvt[tt == 1])) == 0

    def _is_test_vector_valid(self, tv):
        if len(tv) <= self.fbs_size:
            return True

        if len(tv) <= self.max_fbs_size:
            tv = np.array(tv)
            tv_start = tv[0:len(tv) - self.fbs_size]
            tv_end = tv[self.fbs_size:]

            # f(x) == -f(x + fbs_size)
            mode1 = np.all(tv_start != tv_end)
            # f(x) == 0 == -f(x + fbs_size)
            mode2 = np.all(tv_start == tv_end) and np.all(0 == tv_start)
            # f(x) == 1 == -f(x + fbs_size)
            mode3 = np.all(tv_start == tv_end) and np.all(1 == tv_start)
            return mode1 or mode2 or mode3

        return False

    def _is_lut_valid(self, tt, mvt):
        if not self._is_mvt_valid(tt, mvt):
            return False
        if self._mvt_size(mvt) <= self.fbs_size:
            return True

        tt_lut = self._comp_boot_test_vector(tt, mvt, 0)
        if self._is_test_vector_valid(tt_lut):
            return True
        tt_lut = self._comp_boot_test_vector(tt, mvt, 1)
        if self._is_test_vector_valid(tt_lut):
            return True

        return False

    def _get_fbs_test_vector(self, tt, mvt):
        tv = self._comp_boot_test_vector(tt, mvt, 0)
        if self._is_test_vector_valid(tv): return tv
        tv = self._comp_boot_test_vector(tt, mvt, 1)
        assert(self._is_test_vector_valid(tv))
        return tv

    def map_internal(self, env: BitExecEnv, nodes_to_bootstrap):
        logger = self.logger.getChild("map_internal")

        lut_env = LutExecEnv()

        wires = {"0": self.new_const(0), "1": self.new_const(1)}
        for instr in env.instructions:
            match instr:
                case BitExecEnv.Const(name=name):
                    wire = wires[name]
                case BitExecEnv.Input(name=name):
                    wire = self.new_input(lut_env, name)
                case BitExecEnv.LUT(name=name, inputs=inputs, truth_table=truth_table):
                    assert(len(inputs) <=
                           2), "only 1 or 2 input gates are supported"
                    input_wires = list(
                        map(lambda inp: wires[inp.name], inputs))

                    if len(input_wires) == 1:
                        logger.info(f"Gate: {name} <- {inputs[0].name} ({input_wires[0].name()})")
                    else:
                        logger.info(f"Gate: {name} <- {inputs[0].name} {inputs[1].name} ({input_wires[0].name()} {input_wires[1].name()})")

                    wire, bw = self.treat_bit_exec_lut_gate(
                        lut_env, input_wires, truth_table)
                    logger.info(f"\t{wire.name()} {wire.tt} {wire.mvt}")

                    for inp_idx, inp_new_wire in bw.items():
                        be_name = inputs[inp_idx].name
                        logger.info(f"\t\t{be_name} {inp_new_wire.name()}")
                        wires[be_name] = inp_new_wire

                case _:
                    assert(False), "Unknown instruction"

            name = instr.name
            if name in nodes_to_bootstrap:
                logging.info(f"Force node {name} bootstrapping")
                wires[name] = self.new_bootstrap(lut_env, wire)
            else:
                wires[name] = wire

        for name, out in env.outputs.items():
            logging.info(f"Output {name} - {out}")
            node = self.new_output(lut_env, wires[out.name])
            lut_env.output(name, node)

        return lut_env

    def map(self, env: BitExecEnv):
        nodes_to_bootstrap = set(map(lambda e: e.name, env.outputs.values()))
        lut_env = self.map_internal(env, nodes_to_bootstrap)
        return lut_env

    def new_cone(self, *args, **kwargs):
        class Cone:
            def __init__(self, parent, support, coefs, tt, mvt):
                """
                Circuit cone object.

                :param      support:    The cone support
                :type       support:    list of LUT env nodes
                :param      coefs:  The cone coefficients, one per element in support
                :type       coefs:  list of integers
                :param      const_coef:  The constant coefficient
                :type       const_coef:  integer
                :param      tt:          cone truth table
                :type       tt:          list of booleans
                :param      mvt:         cone multi-val table
                :type       mvt:         list of integers
                """
                self.parent = parent
                self.support = np.array(support)
                self.coefs = np.array(coefs)
                self.tt = np.array(tt)
                self.mvt = np.array(mvt)
                assert(self.parent._is_lut_valid(self.tt, self.mvt)), f"{self.tt} {self.mvt}"
                self._support_names = np.array(
                    list(map(lambda e: e.name, self.support)))
                if self.size() != len(np.unique(self.mvt)):
                    logging.critical(f"Cone with sparse mvt: {self.size()} {len(np.unique(self.mvt))}\t{self}")

            def size(self):
                return self.parent._mvt_size(self.mvt)

            def norm2_squared(self):
                return np.sum(self.coefs * self.coefs)

            def name(self):
                return self.__repr__()

            def support_names(self):
                return self._support_names

            def clone_new_tt(self, new_tt):
                return self.parent.new_cone(
                    support=self.support,
                    coefs=self.coefs,
                    tt=new_tt,
                    mvt=self.mvt)

            def is_const(self):
                return len(self.support) == 0

            def __repr__(self):
                return f"Cone({self.support_names()}, {self.coefs}, {self.mvt}, {self.tt})"

        cone = Cone(self, *args, **kwargs)
        # print(f"{wire.name()} {wire.norm2}")
        return cone

    def new_const(self, cst):
        return self.new_cone(
            support=[],
            coefs=[],
            tt=[cst],
            mvt=[0])

    def new_input(self, lut_env, name):
        return self.new_cone(
            support=[lut_env.input(name)],
            coefs=[1],
            tt=[0, 1],
            mvt=[0, 1])

    def _negate(self, cone):
        return cone.clone_new_tt(1 - cone.tt)

    def new_output(self, lut_env, cone):
        if len(cone.support) == 0:  # constant
            return lut_env.const(cone.tt[0])

        if len(cone.support) == 1:  # 1-input cone
            val = cone.support[0]
            if np.all(cone.tt == [1, 0]):  # negate using a linear combination
                return lut_env.linear([-1], [val], const_coef=1)
            else:
                return val

        return self.new_bootstrap(lut_env.cone).support[0]

    def new_bootstrap(self, lut_env, cone):
        if len(cone.support) == 0 or len(cone.support) == 1:  # if cone is constant or has 1-input do nothing
            return cone

        # shift mvt to start from 0
        const_coef = -cone.mvt.min()
        cone.mvt += const_coef

        # linear combination of cone support
        lincomb = lut_env.linear(
            list(map(int, cone.coefs)), cone.support, const_coef=const_coef)

        # bootstrap the lincomb
        lut_tt = self._get_fbs_test_vector(cone.tt, cone.mvt)
        new_val = lut_env.bootstrap(lincomb, lut_tt)
        return self.new_cone(
            support=[new_val],
            coefs=[1],
            tt=[0, 1],
            mvt=[0, 1])

    def _create_and_simplify_cone(self, support, coefs, tt, mvt):
        # remove zero coefficients
        if np.sum(coefs == 0) > 0:
            r = np.zeros(1 << len(coefs), dtype=np.uint32)
            for pos, coef in enumerate(coefs):
                r <<= 1
                if coef != 0:
                    r += self._get_tt_var(len(coefs), pos)
            idx = np.unique(r)

            support = support[coefs != 0]
            coefs = coefs[coefs != 0]

            tt = tt[idx]
            mvt = mvt[idx]

        # reduce coefficients by their gcd
        c = np.gcd.reduce(coefs)
        coefs //= c
        mvt = mvt // c

        return self.new_cone(
            support=support,
            coefs=coefs,
            tt=tt,
            mvt=mvt)

    def _merge_cones(self, cone1, cone2, ab, new_tt, new_mvt):
        cone_sup1 = cone1.support_names()
        cone_sup2 = cone2.support_names()
        a, b = ab

        cone_coefs1_new = cone1.coefs * a
        cone_coefs2_new = cone2.coefs * b

        # move common nodes coefficients to 1st cone
        common_nodes = list(set(cone_sup1).intersection(cone_sup2))
        for node in common_nodes:
            idx1 = np.where(node == cone_sup1)[0][0]
            idx2 = np.where(node == cone_sup2)[0][0]
            cone_coefs1_new[idx1] += cone_coefs2_new[idx2]

        keep_idx = ~np.isin(cone_sup2, common_nodes)

        new_support = np.hstack((cone1.support, cone2.support[keep_idx]))
        new_coefs = np.hstack(
            (cone_coefs1_new, cone_coefs2_new[keep_idx].astype(np.int64)))

        return self._create_and_simplify_cone(new_support, new_coefs, new_tt, new_mvt)

    def _find_lincomb_coefs_naive(self, xy_mvt, r_tt):
        a, b = self._mvt_size(xy_mvt[:, 1]), 1
        r_mvt = a * xy_mvt[:, 0] + b * xy_mvt[:, 1]
        if self._is_lut_valid(r_tt, r_mvt):
            return (a, b), r_mvt
        else:
            return None, None

    def _find_lincomb_coefs_search(self, xy_mvt, r_tt):
        r_ab = None
        r_mvt = None
        r_mvt_max = 1000000000
        r_norm2 = 1000000000

        cone1_mvt_max = self._mvt_size(xy_mvt[:, 0]) - 1
        cone2_mvt_max = self._mvt_size(xy_mvt[:, 1]) - 1

        to_try = set([(cone2_mvt_max + 1, cone1_mvt_max + 1)])
        tried = set()
        while to_try:
            a, b = to_try.pop()
            tried.add((a, b))

            ab = None
            if a > 1:
                ab = (a-1, b)
            elif a == 1:
                ab = (-1, b)
            elif a >= -cone2_mvt_max:
                ab = (a-1, b)
            if ab and ab not in tried:
                to_try.add(ab)

            ab = None
            if b > 1:
                ab = (a, b-1)
            if ab and ab not in tried:
                to_try.add(ab)

            mvt_max = abs(a) * cone1_mvt_max + abs(b) * cone2_mvt_max

            norm2 = a * a + b * b  # approximation of output cone lincomb norm2
            if mvt_max < r_mvt_max or (mvt_max == r_mvt_max and norm2 < r_norm2):
                mvt = a * xy_mvt[:, 0] + b * xy_mvt[:, 1]
                if self._is_lut_valid(r_tt, mvt):
                    r_ab = (a, b)
                    r_mvt = mvt
                    r_mvt_max = mvt_max
                    r_norm2 = norm2

        return r_ab, r_mvt

    def _find_lincomb_coefs_cached(self, xy_mvt, r_tt):
        key1 = f"{','.join(map(str,xy_mvt))}"
        key2 = f"{','.join(map(str,r_tt))}"
        key = f"{key1}|{key2}"
        if key not in self._find_lincomb_coefs_cache:
            self._find_lincomb_coefs_cache[key] = self._find_lincomb_coefs(
                xy_mvt, r_tt)
        return self._find_lincomb_coefs_cache[key]

    def _switch_gate_input_cones(self, cone1, cone2, idx1, idx2, truth_table):
        truth_table[1], truth_table[2] = truth_table[2], truth_table[1]
        return cone2, cone1, idx2, idx1, truth_table

    def _get_tt_var(self, nb_inps, pos_inp):
        # memoize
        assert(pos_inp < nb_inps)
        nb_rep = 1 << (nb_inps - pos_inp - 1)
        seg = np.hstack((np.zeros(nb_rep, dtype=np.uint32),
                         np.ones(nb_rep, dtype=np.uint32)))
        return np.tile(seg, 1 << pos_inp)

    def _get_cone_xy_indices(self, sup1, sup2):
        sup1 = np.array(sup1)
        sup2 = np.array(sup2)

        sup_out = np.concatenate((sup1, sup2[~np.isin(sup2, sup1)]))
        n = len(sup_out)

        idx2 = np.zeros(1 << n, dtype=np.uint32)
        for node in sup2:
            pos = np.where(np.equal(node, sup_out))[0][0]
            idx2 = (idx2 << 1) + self._get_tt_var(n, pos)

        n1 = len(sup1)
        n2 = len(sup2)
        idx1 = np.repeat(np.arange(1 << n1), 1 << (n - n1))

        return idx1, idx2

    def _get_cone_xy_mvt_and_tt(self, cone1, cone2, gate_tt):
        idx1, idx2 = self._get_cone_xy_indices(
            cone1.support_names(), cone2.support_names())

        xy_mvt = np.vstack((cone1.mvt[idx1], cone2.mvt[idx2])).T
        r_tt = np.array(gate_tt)[2 * cone1.tt[idx1] + cone2.tt[idx2]]

        return xy_mvt, r_tt

    def treat_bit_exec_lut_gate(self, lut_env, input_wires, truth_table):
        logger = self.logger.getChild("treat_bit_exec_lut_gate")

        if len(input_wires) == 1:
            cone, = input_wires
            logger.info(f"1-input gate:")
            logger.info(f"\tgate tt: {truth_table}")
            logger.info(f"\tcone support: {list(map(lambda e: e.name, cone.support))}")
            logger.info(f"\t{cone.name()} ({len(cone.tt)} {cone.size()})")
            logger.info(f"\t\t{cone.tt}")
            logger.info(f"\t\t{cone.mvt}")
            assert(len(truth_table) == 2), "error"
            r_tt = np.array(truth_table)[cone.tt]
            logger.info(f"\toutput tt {r_tt}")
            return cone.clone_new_tt(r_tt), {}

        assert(len(input_wires) == 2), "error"
        assert(len(truth_table) == 4), "error"

        cone1, cone2 = input_wires

        logger.info(f"2-input gate:")
        logger.info(f"\tgate tt: {truth_table}")
        logger.info(f"\tcone 1 support: {list(map(lambda e: e.name, cone1.support))}")
        logger.info(f"\t{cone1.name()} ({len(cone1.tt)} {cone1.size()})")
        logger.info(f"\t\t{cone1.tt}")
        logger.info(f"\t\t{cone1.mvt}")
        logger.info(f"\tcone 2 support: {list(map(lambda e: e.name, cone2.support))}")
        logger.info(f"\t{cone2.name()} ({len(cone2.tt)} {cone2.size()})")
        logger.info(f"\t\t{cone2.tt}")
        logger.info(f"\t\t{cone2.mvt}")

        idx1, idx2 = 0, 1
        if cone1.size() < cone2.size() or (cone1.size() == cone2.size() and cone1.norm2_squared() < cone2.norm2_squared()):
            cone1, cone2, idx1, idx2, truth_table = self._switch_gate_input_cones(
                cone1, cone2, idx1, idx2, truth_table)
            logger.info(f"switch cone1 and cone2:")
            logger.info(f"\tgate tt: {truth_table}")

        bootstrapped_wires = dict()

        # force bootstrap if output truth table is too large
        out_tt_size = len(
            set(cone1.support_names()).union(cone2.support_names()))
        if out_tt_size > self.max_truth_table_size:
            logger.info(f"bootstrap {cone1.name()} {idx1}")
            bootstrapped_wires[idx1] = cone1 = self.new_bootstrap(
                lut_env, cone1)
            cone1, cone2, idx1, idx2, truth_table = self._switch_gate_input_cones(
                cone1, cone2, idx1, idx2, truth_table)

            out_tt_size = len(
                set(cone1.support_names()).union(cone2.support_names()))
            if out_tt_size > self.max_truth_table_size:
                logger.info(f"bootstrap {cone1.name()} {idx1}")
                bootstrapped_wires[idx1] = cone1 = self.new_bootstrap(
                    lut_env, cone1)

        xy_mvt, r_tt = self._get_cone_xy_mvt_and_tt(cone1, cone2, truth_table)
        if len(np.unique(r_tt)) == 1:
            return self.new_const(r_tt[0]), bootstrapped_wires

        ab, r_mvt = self._find_lincomb_coefs_cached(xy_mvt, r_tt)

        logger.info(f"optimal lincomb 1st:")
        logger.info(f"\t{ab}")
        logger.info(f"\t{r_tt}")
        logger.info(f"\t{r_mvt}")

        if ab == None:
            logger.info(f"bootstrap {cone1.name()} ({idx1})")
            bootstrapped_wires[idx1] = cone1 = self.new_bootstrap(
                lut_env, cone1)
        else:
            return self._merge_cones(cone1, cone2, ab, r_tt, r_mvt), bootstrapped_wires

        xy_mvt, r_tt = self._get_cone_xy_mvt_and_tt(cone1, cone2, truth_table)
        if len(np.unique(r_tt)) == 1:
            return self.new_const(r_tt[0]), bootstrapped_wires
        ab, r_mvt = self._find_lincomb_coefs_cached(xy_mvt, r_tt)

        logger.info(f"optimal lincomb 2nd:")
        logger.info(f"\t{ab}")
        logger.info(f"\t{r_tt}")
        logger.info(f"\t{r_mvt}")

        if ab == None:
            logger.info(f"bootstrap {cone2.name()} ({idx2})")
            bootstrapped_wires[idx2] = cone2 = self.new_bootstrap(
                lut_env, cone2)
        else:
            return self._merge_cones(cone1, cone2, ab, r_tt, r_mvt), bootstrapped_wires

        xy_mvt, r_tt = self._get_cone_xy_mvt_and_tt(cone1, cone2, truth_table)
        if len(np.unique(r_tt)) == 1:
            return self.new_const(r_tt[0]), bootstrapped_wires
        ab, r_mvt = self._find_lincomb_coefs_cached(xy_mvt, r_tt)

        logger.info(f"optimal lincomb 3rd:")
        logger.info(f"\t{ab}")
        logger.info(f"\t{r_tt}")
        logger.info(f"\t{r_mvt}")

        assert(ab)

        return self._merge_cones(cone1, cone2, ab, r_tt, r_mvt), bootstrapped_wires


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
    lut_env = MapToFBSBasic().map(env)
    lut_env.print()

    print(lut_env.eval(input_vals))
    print()

    print("Naive map:")
    lut_env = MapToFBSHeur(fbs_size=8, max_fbs_size=16,
                           cone_merger="naive").map(env)
    lut_env.print()

    print(lut_env.eval(input_vals))
    print()

    print("Map using exhaustive search:")
    lut_env = MapToFBSHeur(fbs_size=8, max_fbs_size=16,
                           cone_merger="naive").map(env)
    lut_env.print()

    print(lut_env.eval(input_vals))
    print()

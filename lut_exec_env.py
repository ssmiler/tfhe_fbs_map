import sys
import numpy as np
import logging

# -------------------------------------------------------------------------------
# TODO: a.x+b.y == b.y+a.x they must point to same instruction. Sort LinearProd
# coef_vals by vals so that self.instr_cache caches them also, eg.
#


class LutExecEnv:
    class Node:
        def __init__(self, name):
            self.name = name

        def __eq__(self, other):
            return repr(self) == repr(other)

        def __hash__(self):
            return hash(self.name)

    class Const(Node):
        def __init__(self, value):
            super().__init__(f"{value}")
            self.value = value

        def __str__(self):
            return f"{self.value}"

    class Input(Node):
        def __init__(self, name):
            super().__init__(name)

        def __str__(self):
            return f"Input({self.name})"

    class LinearProd(Node):
        def __init__(self, name, coef_vals, const_coef=0):
            super().__init__(name)
            for coef, val in coef_vals:
                assert(isinstance(val, LutExecEnv.Node)
                       ), "Expected 'Node' type"
            self.coef_vals = coef_vals
            self.const_coef = const_coef

        def __str__(self):
            elems = " + ".join(map(lambda cn: f"{cn[0]} * {cn[1].name}", self.coef_vals))
            const_factor = f"+ {self.const_coef}" if self.const_coef != 0 else ""
            return f"{elems} {const_factor}"

    class Bootstrap(Node):
        def __init__(self, name, val, table):
            super().__init__(name)
            assert(isinstance(table, list)), "Expected list"
            assert(isinstance(val, LutExecEnv.Node)
                   ), "Expected LutExecEnv.Node"
            self.table = table
            self.val = val

        def __str__(self):
            return f"Bootstrap({self.val.name}, {self.table})"

    def __init__(self, merge_linear_prods=True):
        self._unique_id = 0
        self.instructions = list()
        self.outputs = dict()
        self._merge_linear_prods = merge_linear_prods
        self.max_val = dict()
        self.instr_cache = dict()
        self.logger = logging.getLogger("LutExecEnv")

    def _new_id(self):
        self._unique_id += 1
        return f"m{self._unique_id}"

    def _set_value_bounds(self, instr):
        assert(instr.name not in self.max_val), "Error"
        match instr:
            case LutExecEnv.Input():
                max_val = 1
            case LutExecEnv.LinearProd(coef_vals=coef_vals, const_coef=const_coef):
                max_val = const_coef + \
                    sum(map(lambda cv: max(
                        0, cv[0] * self.max_val[cv[1].name]), coef_vals))
            case LutExecEnv.Bootstrap(table=table):
                assert(min(table) == 0)
                max_val = max(table)
            case _:
                assert(False), "Unknown instruction"
        self.max_val[instr.name] = max_val
        self.logger.getChild("_set_value_bounds").info(f"{instr.name} {max_val}")

    def _add_instr(self, instr):
        self.logger.getChild("_add_instr").info(f"{instr.name} = {instr}")
        if str(instr) in self.instr_cache:
            return self.instr_cache[str(instr)]
        self.instr_cache[str(instr)] = instr
        self.instructions.append(instr)
        self._set_value_bounds(instr)
        return instr

    def input(self, input_id):
        return self._add_instr(LutExecEnv.Input(input_id))

    def const(self, value):
        return LutExecEnv.Const(value)

    # def linear(self, coefs, vals, const_coef = 0):
    #     val_coef = dict()

    #     for coef, val in zip(coefs, vals):
    #         assert(isinstance(val, LutExecEnv.Node)), "Expected LutExecEnv.Node"
    #         match val:
    #             case LutExecEnv.LinearProd(coef_vals = coef_vals, const_coef = value) if self._merge_linear_prods:
    #                 for coef1, val1 in coef_vals:
    #                     val_coef.setdefault(val1, 0)
    #                     val_coef[val1] += coef * coef1
    #                 const_coef += value
    #             case LutExecEnv.Const(value = value):
    #                 const_coef += value
    #             case _:
    #                 val_coef.setdefault(val, 0)
    #                 val_coef[val] += coef

    #     # filter values with zero coefs
    #     val_coef = filter(lambda e: e[1] != 0, val_coef.items())
    #     res = list(map(lambda e: (e[1], e[0]), val_coef))

    #     return self._add_instr(LutExecEnv.LinearProd(self._new_id(), res, const_coef))

    def linear(self, coefs, vals, const_coef=0):
        res = list()
        for coef, val in zip(coefs, vals):
            assert(isinstance(val, LutExecEnv.Node)
                   ), "Expected LutExecEnv.Node"
            match val:
                case LutExecEnv.LinearProd(coef_vals=coef_vals, const_coef=value) if self._merge_linear_prods:
                    for coef1, val1 in coef_vals:
                        res.append((coef * coef1, val1))
                    const_coef += coef * value
                case LutExecEnv.Const(value=value):
                    const_coef += coef * value
                case _:
                    res.append((coef, val))
        return self._add_instr(LutExecEnv.LinearProd(self._new_id(), res, const_coef))

    def bootstrap(self, val, table):
        assert(isinstance(val, LutExecEnv.Node)), "Expected LutExecEnv.Node"
        assert(isinstance(table, list)), "Expected list"
        assert(len(table) == self.max_val[val.name] + 1), f"{table} vs {val.name} {self.max_val[val.name]}"
        res = self._add_instr(LutExecEnv.Bootstrap(self._new_id(), val, table))
        return res

    def output(self, name, val):
        assert(isinstance(val, LutExecEnv.Node)), "Expected LutExecEnv.Node"
        self.outputs[name] = val

    def print(self, os=sys.stdout, show_inputs=False, show_outputs=False):
        for instr in self.instructions:
            match instr:
                case LutExecEnv.Input() if not show_inputs:
                    pass
                case _:
                    print(f"{instr.name} = {str(instr)}", file=os)

        if show_outputs:
            for name, val in self.outputs.items():
                print(f"Output {name} = {val.name}", file=os)

    def eval(self, input_values):
        wire_values = {"0": 0, "1": 1}

        for instr in self.instructions:
            match instr:
                case LutExecEnv.Input(name=name):
                    val = np.array(input_values[name]).reshape(-1)
                case LutExecEnv.LinearProd(coef_vals=coef_vals, const_coef=const_coef):
                    val = np.sum(list(
                        map(lambda cn: cn[0] * wire_values[cn[1].name], coef_vals)), axis=0) + const_coef
                case LutExecEnv.Bootstrap(val=val, table=table):
                    val = np.fromiter(
                        map(lambda v: table[v], wire_values[val.name]), dtype=int)
                case _:
                    assert(False), "Unknown instruction"
            wire_values[instr.name] = val

        output_values = dict()
        for name, out in self.outputs.items():
            output_values[name] = wire_values[out.name]

        return output_values

    def remove_dangling_nodes(self):
        visited = set(map(lambda out: out.name, self.outputs.values()))

        for instr in self.instructions[::-1]:
            if instr.name in visited:
                match instr:
                    case LutExecEnv.LinearProd(coef_vals=coef_vals):
                        visited.update(map(lambda cv: cv[1].name, coef_vals))
                    case LutExecEnv.Bootstrap(val=val):
                        visited.add(val.name)

        self.instructions = list(
            filter(lambda instr: instr.name in visited, self.instructions))

    def stats(self):
        nb_inp = 0
        nb_linprod = 0
        nb_bootstrap = 0
        max_lut_size = 0

        norm2_vals = dict()
        for instr in self.instructions:
            match instr:
                case LutExecEnv.Input(name=name):
                    nb_inp += 1
                    norm2_vals[name] = 1
                case LutExecEnv.LinearProd(name=name, coef_vals=coef_vals):
                    nb_linprod += 1
                    norm2_vals[name] = sum(
                        map(lambda e: e[0] * e[0] * norm2_vals[e[1].name], coef_vals))
                case LutExecEnv.Bootstrap(name=name, table=table):
                    nb_bootstrap += 1
                    max_lut_size = max(max_lut_size, len(table))
                    norm2_vals[name] = 1
                case _:
                    assert(False), "Unknown instruction"

        d = dict(
            nb_inp=nb_inp,
            nb_linprod=nb_linprod,
            nb_bootstrap=nb_bootstrap,
            max_lut_size=max_lut_size,
            norm2_linprod=max(norm2_vals.values()),
            nb_out=len(self.outputs),
        )
        return d


if __name__ == '__main__':
    env = LutExecEnv()

    a = env.input("a")
    b = env.input("b")
    c = env.const(1)

    d = env.linear([1, 2], [a, b])
    e = env.linear([1, 1], [c, d])
    f = env.bootstrap(e, [1, 0, 1, 1, 0])

    g = env.linear([2, 1], [a, f])
    h = env.bootstrap(g, [1, 1, 0, 2])
    i = env.bootstrap(h, [1, 0, 1])

    env.output("f", f)
    env.output("g", g)
    env.output("h", h)

    env.print()

    # print(env.eval({"a": 1, "b": 1, "c": 0}))
    print(env.eval({"a": [1, 0], "b": [1, 0], "c": [1, 0]}))

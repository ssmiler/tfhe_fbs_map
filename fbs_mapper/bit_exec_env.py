import sys
import numpy as np


class BitExecEnv:
    class Node:
        def __init__(self, name):
            self.name = name

    class Const(Node):
        def __init__(self, val):
            super().__init__(f"CONST{val}")
            self.val = val

        def __str__(self):
            return self.name

    CONST0 = Const(0)
    CONST1 = Const(1)

    class Input(Node):
        def __init__(self, name):
            super().__init__(name)

        def __str__(self):
            return f"Input({self.name})"

    class LUT(Node):
        def __init__(self, name, inputs, truth_table):
            super().__init__(name)
            for inp in inputs:
                assert(isinstance(inp, BitExecEnv.Node)), "something is wrong"
            self.inputs = inputs
            self.truth_table = truth_table

        def __str__(self):
            inputs_str = ", ".join(map(lambda inp: inp.name, self.inputs))
            return f"LUT([{inputs_str}], {self.truth_table})"

    class And(LUT):
        def __init__(self, name, inp1, inp2):
            super().__init__(name, [inp1, inp2], truth_table=[0, 0, 0, 1])
            assert(inp1.name != inp2.name), "something is wrong"

        def __str__(self):
            return f"AND({self.inputs[0].name}, {self.inputs[1].name})"

    class Xor(LUT):
        def __init__(self, name, inp1, inp2):
            super().__init__(name, [inp1, inp2], truth_table=[0, 1, 1, 0])
            assert(inp1.name != inp2.name), "something is wrong"

        def __str__(self):
            return f"XOR({self.inputs[0].name}, {self.inputs[1].name})"

    class Or(LUT):
        def __init__(self, name, inp1, inp2):
            super().__init__(name, [inp1, inp2], truth_table=[0, 1, 1, 1])
            assert(inp1.name != inp2.name), "something is wrong"

        def __str__(self):
            return f"OR({self.inputs[0].name}, {self.inputs[1].name})"

    class Not(LUT):
        def __init__(self, name, inp):
            super().__init__(name, [inp], truth_table=[1, 0])

        def __str__(self):
            return f"Not({self.inputs[0].name})"

    def __init__(self):
        self._unique_id = 0
        self.instructions = list()
        self.inputs = list()
        self.outputs = dict()
        self.ids = set()

    def _new_id(self):
        self._unique_id += 1
        return f"n{self._unique_id}"

    def _get_id(self, name):
        if name is None:
            name = self._new_id()
            while name in self.ids:
                name = self._new_id()
        else:
            assert(name not in self.ids), "id already exists in circuit"
        self.ids.add(name)
        return name

    def _add_instr(self, instr):
        self.instructions.append(instr)
        return instr

    def input(self, input_id):
        inp = self._add_instr(BitExecEnv.Input(input_id))
        self.inputs.append(inp)
        return inp

    def output(self, name, node):
        assert(isinstance(node, BitExecEnv.Node)), "Expected BitExecEnv.Node"
        self.outputs[name] = node

    def op_lut(self, inputs, truth_table, name=None):
        assert(2 ** len(inputs) == len(truth_table)), "length miss-match"
        for inp in inputs:
            assert(isinstance(inp, BitExecEnv.Node)), "Error"
        assert(min(truth_table) == 0), "truth table wrong values"
        assert(max(truth_table) == 1), "truth table wrong values"
        return self._add_instr(BitExecEnv.LUT(self._get_id(name), inputs, truth_table))

    def op_not(self, inp, name=None):
        match inp:
            case BitExecEnv.CONST0:
                return BitExecEnv.CONST1
            case BitExecEnv.CONST1:
                return BitExecEnv.CONST0
            case inp:
                return self._add_instr(BitExecEnv.Not(self._get_id(name), inp))

    def op_and(self, inp1, inp2, name=None):
        match(inp1, inp2):
            case(BitExecEnv.CONST0, _):
                return BitExecEnv.CONST0
            case(BitExecEnv.CONST1, inp):
                return inp
            case(_, BitExecEnv.CONST0):
                return BitExecEnv.CONST0
            case(inp, BitExecEnv.CONST1):
                return inp
            case(inp1, inp2):
                return self._add_instr(BitExecEnv.And(self._get_id(name), inp1, inp2))

    def op_xor(self, inp1, inp2, name=None):
        match(inp1, inp2):
            case(BitExecEnv.CONST0, inp):
                return inp
            case(BitExecEnv.CONST1, inp):
                return self.op_not(inp)
            case(inp, BitExecEnv.CONST0):
                return inp
            case(inp, BitExecEnv.CONST1):
                return self.op_not(inp)
            case(inp1, inp2):
                return self._add_instr(BitExecEnv.Xor(self._get_id(name), inp1, inp2))

    def op_or(self, inp1, inp2, name=None):
        match(inp1, inp2):
            case(BitExecEnv.CONST0, inp):
                return inp
            case(BitExecEnv.CONST1, inp):
                return BitExecEnv.CONST1
            case(inp, BitExecEnv.CONST0):
                return inp
            case(inp, BitExecEnv.CONST1):
                return BitExecEnv.CONST1
            case(inp1, inp2):
                return self._add_instr(BitExecEnv.Or(self._get_id(name), inp1, inp2))

    def print(self, os=sys.stdout, show_inputs=True, show_outputs=True):
        for instr in self.instructions:
            match instr:
                case BitExecEnv.Input() if not show_inputs:
                    pass
                case _:
                    print(f"{instr.name} = {str(instr)}", file=os)

        if show_outputs:
            for name, out in self.outputs.items():
                print(f"Output {name} = {out.name}", file=os)

    def eval(self, input_values):
        wire_values = {BitExecEnv.CONST0.name: 0, BitExecEnv.CONST1.name: 1}

        for instr in self.instructions:
            match instr:
                case BitExecEnv.Const(name=val):
                    pass
                case BitExecEnv.Input(name=name):
                    val = np.array(input_values[name]).reshape(-1)
                case BitExecEnv.LUT(name=name, inputs=inputs, truth_table=table):
                    idx = sum(
                        map(lambda e: wire_values[e[1].name] * 2**e[0], enumerate(inputs[::-1])))
                    val = np.fromiter(map(lambda v: table[v], idx), dtype=int)
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
                    case BitExecEnv.LUT(name=name, inputs=inputs):
                        visited.update(map(lambda inp: inp.name, inputs))

        self.instructions = list(
            filter(lambda instr: instr.name in visited, self.instructions))

    def stats(self):
        nb_inp = 0
        nb_and = 0
        nb_xor = 0
        nb_not = 0
        nb_lut = 0
        max_lut_inputs = 0
        max_lut_size = 0

        norm2_vals = dict()
        for instr in self.instructions:
            match instr:
                case BitExecEnv.Input():
                    nb_inp += 1
                case BitExecEnv.And():
                    nb_and += 1
                case BitExecEnv.Xor():
                    nb_xor += 1
                case BitExecEnv.Not():
                    nb_not += 1
                case BitExecEnv.LUT(inputs=inputs, truth_table=truth_table):
                    nb_lut += 1
                    max_lut_inputs = max(max_lut_inputs, len(inputs))
                    max_lut_size = max(max_lut_size, len(truth_table))
                case _:
                    assert(False), "Unknown instruction"

        d = dict(
            nb_inp=nb_inp,
            nb_and=nb_and,
            nb_xor=nb_xor,
            nb_not=nb_not,
            nb_lut=nb_lut,
            max_lut_inputs=max_lut_inputs,
            max_lut_size=max_lut_size,
            nb_out=len(self.outputs),
        )
        return d

    def to_blif(self, fs=sys.stdout, model_name="test"):
        def to_blif_table(truth_table):
            val = 1 if np.mean(truth_table) <= 0.5 else 0
            l = int(np.log2(len(truth_table)))
            fmt_str = f"{{:0{l}b}} {{}}"
            indices = map(lambda e: e[0], filter(
                lambda e: e[1] == val, enumerate(truth_table)))
            return "\n".join(map(lambda idx: fmt_str.format(idx, val), indices))

        print(f".model {model_name}", file=fs)

        print(f".inputs {' '.join(map(lambda inp: inp.name, self.inputs))}", file=fs)

        print(f".outputs {' '.join(self.outputs.keys())}", file=fs)

        for instr in self.instructions:
            match instr:
                case BitExecEnv.Const(name=val):
                    print(f".names CONST{val}", file=fs)
                    print(f"{val}", file=fs)
                case BitExecEnv.Input(name=name):
                    pass
                case BitExecEnv.LUT(name=name, inputs=inputs, truth_table=table):
                    print(f".names {' '.join(map(lambda inp: inp.name, inputs))} {name}", file=fs)
                    print(to_blif_table(table), file=fs)
                case _:
                    assert(False), "Unknown instruction"

        for name, out in self.outputs.items():
            if out.name != name:
                print(f".names {out.name} {name}\n1 1", file=fs)

        print('.end', file=fs)


if __name__ == '__main__':
    import itertools as it

    env = BitExecEnv()

    a = env.input("a")
    b = env.CONST0
    # b = env.input("b")

    d = env.op_and(a, b, "n11")
    e = env.op_xor(a, b)
    f = env.op_lut([a, b], [0, 1, 0, 0])

    env.output("d", d)
    env.output("e", e)
    env.output("f", f)

    env.print()

    input_vals = {"a": [0, 0, 1, 1], "b": [0, 1, 0, 1]}
    print(env.eval(input_vals))

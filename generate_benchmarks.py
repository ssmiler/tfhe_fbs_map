import argparse
from bit_exec_env import *


class Bit:
    @classmethod
    def set_env(cls, env):
        Bit.env = env

    def __init__(self, val):
        self.val = val

    @classmethod
    def input(self, name):
        return Bit(Bit.env.input(name))

    @classmethod
    def const(self, val):
        return Bit(Bit.env.CONST1 if val else Bit.env.CONST0)

    def output(self, name=None):
        Bit.env.output(name if name else self.val.name, self.val)
        return self

    def __or__(self, other):
        return Bit(Bit.env.op_or(self.val, other.val))

    def __xor__(self, other):
        return Bit(Bit.env.op_xor(self.val, other.val))

    def __and__(self, other):
        return Bit(Bit.env.op_and(self.val, other.val))

    def __invert__(self):
        return Bit(Bit.env.op_not(self.val))


def full_adder_bench():
    a = Bit.input("a")
    b = Bit.input("b")
    c = Bit.input("cin")

    n1 = c ^ a;
    n2 = c ^ b;
    y = n1 ^ b;
    cout = (n1 & n2) ^ c;

    y.output("out")
    cout.output("cout")
full_adder_bench.__name__ = "full_adder"

class KreyviumIter:
    def iter_v1(s, K127, IV127):
        t1 = s[66] ^ s[93]
        t2 = s[162] ^ s[177]
        t3 = s[243] ^ s[288] ^ K127

        r = t1 ^ t2 ^ t3

        t1 = t1 ^ (s[91] & s[92]) ^ s[171] ^ IV127
        t2 = t2 ^ (s[175] & s[176]) ^ s[264]
        t3 = t3 ^ (s[286] & s[287]) ^ s[69]
        return r, t1, t2, t3

    def iter_v2(s, K127, IV127):
        t1 = s[66] ^ s[93]
        t2 = s[162] ^ s[177]
        t3 = s[243] ^ s[288] ^ K127

        r = t1 ^ t2 ^ t3

        t1 = (t1 ^ s[171] ^ IV127) ^ (s[91] & s[92])
        t2 = (t2 ^ s[264]) ^ (s[175] & s[176])
        t3 = (t3 ^ s[69]) ^ (s[286] & s[287])
        return r, t1, t2, t3

    def iter_v3(s, K127, IV127):
        t1 = s[66] ^ s[93]
        t2 = s[162] ^ s[177]
        t3 = s[243] ^ s[288] ^ K127

        r = t1 ^ t2 ^ t3

        t1 = (s[91] & s[92]) ^ (t1 ^ s[171] ^ IV127)
        t2 = (s[175] & s[176]) ^ (t2 ^ s[264])
        t3 = (s[286] & s[287]) ^ (t3 ^ s[69])
        return r, t1, t2, t3

    def kreyvium_iter(iter_fnc):
        input_indices = [66, 162, 243, 91, 92, 93,
                         175, 176, 177, 286, 287, 288, 69, 171, 264]
        s = dict([(k, Bit.input(f"s{k}")) for k in input_indices])
        K127 = Bit.input(f"k127")
        IV127 = Bit.input(f"IV127")

        r, t1, t2, t3 = iter_fnc(s, K127, IV127)

        r.output("y")
        t1.output("t1")
        t2.output("t2")
        t3.output("t3")

    def kreyvium_iter_v1():
        KreyviumIter.kreyvium_iter(KreyviumIter.iter_v1)

    def kreyvium_iter_v2():
        KreyviumIter.kreyvium_iter(KreyviumIter.iter_v2)

    def kreyvium_iter_v3():
        KreyviumIter.kreyvium_iter(KreyviumIter.iter_v3)


class TriviumIter:
    def iter_v1(s):
        t1 = s[66] ^ s[93]
        t2 = s[162] ^ s[177]
        t3 = s[243] ^ s[288]

        r = t1 ^ t2 ^ t3

        t1 = t1 ^ (s[91] & s[92]) ^ s[171]
        t2 = t2 ^ (s[175] & s[176]) ^ s[264]
        t3 = t3 ^ (s[286] & s[287]) ^ s[69]
        return r, t1, t2, t3

    def iter_v2(s):
        t1 = s[66] ^ s[93]
        t2 = s[162] ^ s[177]
        t3 = s[243] ^ s[288]

        r = t1 ^ t2 ^ t3

        t1 = (t1 ^ s[171]) ^ (s[91] & s[92])
        t2 = (t2 ^ s[264]) ^ (s[175] & s[176])
        t3 = (t3 ^ s[69]) ^ (s[286] & s[287])
        return r, t1, t2, t3

    def iter_v3(s):
        t1 = s[66] ^ s[93]
        t2 = s[162] ^ s[177]
        t3 = s[243] ^ s[288]

        r = t1 ^ t2 ^ t3

        t1 = (s[91] & s[92]) ^ (t1 ^ s[171])
        t2 = (s[175] & s[176]) ^ (t2 ^ s[264])
        t3 = (s[286] & s[287]) ^ (t3 ^ s[69])
        return r, t1, t2, t3

    def trivium_iter(iter_fnc):
        input_indices = [66, 162, 243, 91, 92, 93,
                         175, 176, 177, 286, 287, 288, 69, 171, 264]
        s = dict([(k, Bit.input(f"s{k}")) for k in input_indices])

        r, t1, t2, t3 = iter_fnc(s)

        r.output("y")
        t1.output("t1")
        t2.output("t2")
        t3.output("t3")

    def trivium_iter_v1():
        TriviumIter.trivium_iter(TriviumIter.iter_v1)

    def trivium_iter_v2():
        TriviumIter.trivium_iter(TriviumIter.iter_v2)

    def trivium_iter_v3():
        TriviumIter.trivium_iter(TriviumIter.iter_v3)


class TriviumState:
    def trivium_state(iter_fnc):
        s = [None]
        s.extend([Bit.input(f"K{k-1}") for k in range(1, 1 + 80)])
        s.extend([Bit.const(0) for k in range(1 + 80, 94)])
        s.extend([Bit.input(f"IV{k-94}") for k in range(94, 94 + 80)])
        s.extend([Bit.const(0) for k in range(94 + 80, 286)])
        s.extend([Bit.const(1) for k in range(286, 289)])

        for i in range(1152):
            r, t1, t2, t3 = iter_fnc(s)

            s[1:94] = [t3, *s[1:93]]
            s[94:178] = [t1, *s[94:177]]
            s[178:289] = [t2, *s[178:288]]

        for i in range(1, 289):
            s[i].output(f"s{i}")

    def trivium_state_v1():
        TriviumState.trivium_state(TriviumIter.iter_v1)

    def trivium_state_v2():
        TriviumState.trivium_state(TriviumIter.iter_v2)

    def trivium_state_v3():
        TriviumState.trivium_state(TriviumIter.iter_v3)


class TriviumStream:
    def trivium_stream(iter_fnc):
        s = [None]
        s.extend([Bit.input(f"s{k}") for k in range(1, 289)])

        iters = 288*4
        r = []
        for i in range(iters):
            r, t1, t2, t3 = iter_fnc(s)
            r.output(f"r{i}")

            s[1:94] = [t3, *s[1:93]]
            s[94:178] = [t1, *s[94:177]]
            s[178:289] = [t2, *s[178:288]]

        for i in range(1, 289):
            s[i].output()

    def trivium_stream_v1():
        TriviumStream.trivium_stream(TriviumIter.iter_v1)

    def trivium_stream_v2():
        TriviumStream.trivium_stream(TriviumIter.iter_v2)

    def trivium_stream_v3():
        TriviumStream.trivium_stream(TriviumIter.iter_v3)


class KreyviumStream:
    def kreyvium_stream(iter_fnc):
        s = [None]
        s.extend([Bit.input(f"s{k}") for k in range(1, 289)])

        K = [Bit.input(f"K{k}") for k in range(128)]
        IV = [Bit.input(f"IV{k}") for k in range(128)]

        iters = 288*4
        r = []
        for i in range(iters):
            r, t1, t2, t3 = iter_fnc(s, K[127], IV[127])
            r.output(f"r{i}")

            s[1:94] = [t3, *s[1:93]]
            s[94:178] = [t1, *s[94:177]]
            s[178:289] = [t2, *s[178:288]]
            K = [K[127], *K[:127]]
            IV = [IV[127], *IV[:127]]

        for i in range(1, 289):
            s[i].output()

    def kreyvium_stream_v1():
        KreyviumStream.kreyvium_stream(KreyviumIter.iter_v1)

    def kreyvium_stream_v2():
        KreyviumStream.kreyvium_stream(KreyviumIter.iter_v2)

    def kreyvium_stream_v3():
        KreyviumStream.kreyvium_stream(KreyviumIter.iter_v3)


if __name__ == '__main__':
    bench_generators_fnc = [
                            # full_adder_bench,
                            KreyviumIter.kreyvium_iter_v2,
                            KreyviumIter.kreyvium_iter_v3,
                            TriviumIter.trivium_iter_v1,
                            TriviumIter.trivium_iter_v2,
                            TriviumIter.trivium_iter_v3,
                            KreyviumStream.kreyvium_stream_v1,
                            KreyviumStream.kreyvium_stream_v2,
                            KreyviumStream.kreyvium_stream_v3,
                            TriviumStream.trivium_stream_v1,
                            TriviumStream.trivium_stream_v2,
                            TriviumStream.trivium_stream_v3,
                            # TriviumState.trivium_state_v1,
                            # TriviumState.trivium_state_v2,
                            # TriviumState.trivium_state_v3,
                            ]
    bench_generators = dict(
        map(lambda fnc: (fnc.__name__, fnc), bench_generators_fnc))

    parser = argparse.ArgumentParser(
        prog="Generate benchmark blifs",
        description="Map logic gates to Functional Boostrapping (FBS)")
    parser.add_argument("--prefix", help="output directory", required=True)

    args = parser.parse_args()

    for bench, gen in bench_generators.items():
        bit_env = BitExecEnv()
        Bit.set_env(bit_env)

        gen()

        bit_env.remove_dangling_nodes()
        out_filename = f"{args.prefix}/{bench}.blif"
        fs = open(out_filename, "w")
        bit_env.to_blif(fs=fs, model_name=bench)

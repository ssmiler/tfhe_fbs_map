import argparse
import context
from fbs_mapper.bit_exec_env import *


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

    n1 = c ^ a
    n2 = c ^ b
    out = n1 ^ b
    cout = (n1 & n2) ^ c

    out.output("out")
    cout.output("cout")

full_adder_bench.__name__ = "full_adder"


def ascon_lut():
    x0, x1, x2, x3, x4 = [Bit.input(f"x{k}") for k in range(5)]

    x0 = x0 ^ x4
    x2 = x1 ^ x2
    x4 = x3 ^ x4

    l2_0 = ~x0 & x1
    l2_1 = ~x1 & x2
    l2_2 = ~x2 & x3
    l2_3 = ~x3 & x4
    l2_4 = ~x4 & x0

    x0 = x0 ^ l2_1
    x1 = x1 ^ l2_2
    x2 = x2 ^ l2_3
    x3 = x3 ^ l2_4
    x4 = x4 ^ l2_0

    x1 = x0 ^ x1
    x3 = x2 ^ x3
    x0 = x0 ^ x4
    x2 = ~x2

    x0.output("x0")
    x1.output("x1")
    x2.output("x2")
    x3.output("x3")
    x4.output("x4")
ascon_lut.__name__ = "ascon_lut"


def aes_sbox():
    y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17, y18, y19, y20, y21 = [Bit.input(f"y{k}") for k in range(1, 22)]
    x7 = Bit.input("x7")

    t2 = y12 & y15
    t3 = y3 & y6
    t4 = t3 ^ t2
    t5 = y4 & x7
    t6 = t5 ^ t2
    t7 = y13 & y16
    t8 = y5 & y1
    t9 = t8 ^ t7
    t10 = y2 & y7
    t11 = t10 ^ t7
    t12 = y9 & y11
    t13 = y14 & y17
    t14 = t13 ^ t12
    t15 = y8 & y10
    t16 = t15 ^ t12
    t17 = t4 ^ t14
    t18 = t6 ^ t16
    t19 = t9 ^ t14
    t20 = t11 ^ t16
    t21 = t17 ^ y20
    t22 = t18 ^ y19
    t23 = t19 ^ y21
    t24 = t20 ^ y18
    t25 = t21 ^ t22
    t26 = t21 & t23
    t27 = t24 ^ t26
    t28 = t25 & t27
    t29 = t28 ^ t22
    t30 = t23 ^ t24
    t31 = t22 ^ t26
    t32 = t31 & t30
    t33 = t32 ^ t24
    t34 = t23 ^ t33
    t35 = t27 ^ t33
    t36 = t24 & t35
    t37 = t36 ^ t34
    t38 = t27 ^ t36
    t39 = t29 & t38
    t40 = t25 ^ t39
    t41 = t40 ^ t37
    t42 = t29 ^ t33
    t43 = t29 ^ t40
    t44 = t33 ^ t37
    t45 = t42 ^ t41
    (t44 & y15).output("z0")
    (t37 & y6).output("z1")
    (t33 & x7).output("z2")
    (t43 & y16).output("z3")
    (t40 & y1).output("z4")
    (t29 & y7).output("z5")
    (t42 & y11).output("z6")
    (t45 & y17).output("z7")
    (t41 & y10).output("z8")
    (t44 & y12).output("z9")
    (t37 & y3).output("z10")
    (t33 & y4).output("z11")
    (t43 & y13).output("z12")
    (t40 & y5).output("z13")
    (t29 & y2).output("z14")
    (t42 & y9).output("z15")
    (t45 & y14).output("z16")
    (t41 & y8).output("z17")

aes_sbox.__name__ = "aes_sbox"


def half_adder_bench():
    a = Bit.input("a")
    b = Bit.input("b")

    out = a ^ b
    cout = a & b

    out.output("out")
    cout.output("cout")

half_adder_bench.__name__ = "half_adder"


def simon_iter():
    b0 = Bit.input("b0")
    b1 = Bit.input("b1")
    b2 = Bit.input("b2")
    b3 = Bit.input("b3")
    b4 = Bit.input("b4")

    out = (b0 & b1) ^ b2 ^ b3 ^ b4

    out.output("out")

simon_iter.__name__ = "simon_iter"


def _2_input_gates():
    a = Bit.input("a")
    b = Bit.input("b")

    _and = a & b
    _nand = ~(a & b)
    _andyn = a & ~b
    _andnu = ~a & b

    _or = a | b
    _nor = ~(a | b)
    _oryn = a | ~b
    _ornu = ~a | b

    _xor = a ^ b
    _xnor = ~(a ^ b)

    _and.output("and")
    _nand.output("nand")
    _andyn.output("andyn")
    _andnu.output("andnu")
    _or.output("or")
    _nor.output("nor")
    _oryn.output("oryn")
    _ornu.output("ornu")
    _xor.output("xor")
    _xnor.output("xnor")


_2_input_gates.__name__ = "_2_input_gates"


def aoi21_bench():
    a = Bit.input("a")
    b = Bit.input("b")
    c = Bit.input("c")

    out = ~((a & b) | c)

    out.output("out")

aoi21_bench.__name__ = "aoi21"


def oai21_bench():
    a = Bit.input("a")
    b = Bit.input("b")
    c = Bit.input("c")

    out = ~((a | b) & c)

    out.output("out")

oai21_bench.__name__ = "oai21"


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
                            ascon_lut,
                            aes_sbox,
                            simon_iter,
                            _2_input_gates,
                            full_adder_bench,
                            half_adder_bench,
                            aoi21_bench,
                            oai21_bench,
                            KreyviumIter.kreyvium_iter_v1,
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

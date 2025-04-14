from fbs_exec_env import *
from typing import List, Dict, TextIO


class LbfCircuitParser:
    def _strip_comments(self, line):
        # strip comments and spaces
        idx = line.find("#")
        if idx >= 0:
            line = line[:idx]
        line = line.strip()
        return line

    def _get_callback(self, callback):
        def _empty_callback(*args):
            pass

        if callback is None:
            return _empty_callback
        else:
            return callback

    def __init__(self,
                 inputs_callback=None,
                 outputs_callback=None,
                 lincomb_callback=None,
                 const_callback=None,
                 bootstrap_callback=None,
                 end_callback=None):
        self.inputs_callback = self._get_callback(inputs_callback)
        self.outputs_callback = self._get_callback(outputs_callback)
        self.lincomb_callback = self._get_callback(lincomb_callback)
        self.const_callback = self._get_callback(const_callback)
        self.bootstrap_callback = self._get_callback(bootstrap_callback)
        self.end_callback = self._get_callback(end_callback)

    def parse(self, lines):
        idx = 0
        while idx < len(lines):
            line, idx = self._strip_comments(lines[idx]), idx + 1
            match line.split():
                case[".inputs", *inps]:
                    while inps[-1] == "\\":
                        line, idx = self._strip_comments(lines[idx]), idx + 1
                        inps = inps[:-1] + line.split()
                    self.inputs_callback(inps)  # TODO: support multi-line

                case[".outputs", *outs]:
                    while outs[-1] == "\\":
                        line, idx = self._strip_comments(lines[idx]), idx + 1
                        outs = outs[:-1] + line.split()
                    self.outputs_callback(outs)  # TODO: support multi-line

                case[".lincomb", *inps, out]:
                    line, idx = self._strip_comments(lines[idx]), idx + 1
                    coefs = list(map(int, line.split()))

                    const_coef = 0
                    if len(coefs) > len(inps):
                        *coefs, const_coef = coefs
                    assert(len(coefs) == len(inps)
                           ), "lincomb input and coefficient count missmatch"

                    if len(inps) > 0:
                        self.lincomb_callback(out, inps, coefs, const_coef)
                    else:
                        self.const_callback(out, const_coef)

                case[".bootstrap", inp, *outs]:
                    assert(len(outs) > 0), "bootstrap expected at least"
                    " one output"
                    tables = list()
                    for _ in range(len(outs)):
                        line, idx = self._strip_comments(lines[idx]), idx + 1
                        tables.append(list(map(int, line)))
                    self.bootstrap_callback(outs, inp, tables)

                case[".end"]:
                    pass

                case _:
                    assert(False), f"cannot parse line no {idx}: '{line}'"

        self.end_callback()


class LutEnvParser:
    def __init__(self):
        pass

    def parse_file(filename: str):
        with open(filename, "r") as fs:
            return LutEnvParser.parse_stream(fs)

    def parse_stream(fs: TextIO):
        lines = fs.readlines()
        return LutEnvParser.parse_lines(lines)

    def parse_lines(lines: List[str]):
        env = LutExecEnv()
        outputs = list()
        node_val = dict()

        def inputs_callback(inps):
            for inp in inps:
                assert(inp not in node_val)
                node_val[inp] = env.input(inp)

        def outputs_callback(outs):
            outputs.extend(outs)

        def lincomb_callback(out, inps, coefs, const_coef):
            if len(inps) == 1 and coefs[0] == 1 and const_coef == 0:
                node_val[out] = node_val[inps[0]]
            else:
                vals = list(map(lambda inp: node_val[inp], inps))
                node_val[out] = env.linear(coefs, vals, const_coef)

        def const_callback(out, val):
            node_val[out] = env.const(val)

        def bootstrap_callback(outs, inp, tables):
            assert(len(outs) == len(tables))
            val = node_val[inp]
            if len(outs) == 1:
                out, table = outs[0], tables[0]
                node_val[out] = env.bootstrap(val, table)
            else:
                _, bootstraps = env.bootstrap_multi(val, tables)
                for out, bootstrap in zip(outs, bootstraps):
                    node_val[out] = bootstrap

        def end_callback(*args):
            for out in outputs:
                env.output(out, node_val[out])

        parser = LbfCircuitParser(
            inputs_callback=inputs_callback,
            outputs_callback=outputs_callback,
            lincomb_callback=lincomb_callback,
            const_callback=const_callback,
            bootstrap_callback=bootstrap_callback,
            end_callback=end_callback)

        parser.parse(lines)

        return env


if __name__ == '__main__':
    env = LutEnvParser.parse_file("sample.lbf")
    print(env.stats())

    print(env.eval({"a": [1, 0], "b": [1, 0], "c": [1, 0], "d": [1, 1]}))

    env.write_lbf()

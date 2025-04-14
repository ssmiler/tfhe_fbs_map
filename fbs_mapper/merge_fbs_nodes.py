import itertools as it
import numpy as np
import argparse
import copy
import logging

from fbs_exec_env import LutExecEnv
from lbf_parser import LutEnvParser, LbfCircuitParser

powers_of_2 = [2**i for i in range(16)]
bool_val_combs = dict()
fbs_size = 32
max_fbs_size = fbs_size * 2

def comp_mvt(coefs, const_coef):
    t = bool_val_combs.setdefault(len(coefs), np.array(list(it.product(*[[0,1]]*len(coefs)))))
    return t @ coefs + const_coef

def is_mvt_valid(tt, mvt):
    return len(set(mvt[tt == 0]).intersection(mvt[tt == 1])) == 0

def is_test_vector_valid(tv):
    if len(tv) <= fbs_size:
        return True

    if len(tv) <= 2 * fbs_size:
        tv = np.array(tv)
        tv_start = tv[0:len(tv) - fbs_size]
        tv_end = tv[fbs_size:]

        # f(x) == -f(x + fbs_size)
        mode1 = np.all(tv_start != tv_end)
        # f(x) == 0 == -f(x + fbs_size)
        mode2 = np.all(tv_start == tv_end) and np.all(0 == tv_start)
        # f(x) == 1 == -f(x + fbs_size)
        mode3 = np.all(tv_start == tv_end) and np.all(1 == tv_start)
        return mode1 or mode2 or mode3

    return False

def comp_fbs_test_vector(tt, mvt, missing_val):
    tmp = dict(it.product(range(mvt.min(), mvt.max() + 1), [missing_val]))
    tmp.update(zip(mvt, tt))
    return list(map(lambda e: e[1], sorted(tmp.items())))

def is_lut_valid(tt, mvt):
    res = []
    for v in [0,1]:
        tt_lut = comp_fbs_test_vector(tt, mvt, v)
        if is_test_vector_valid(tt_lut):
            res.append(v)
    return res

def is_lut_valid_new_mvt(tt_new, tt, mvt):
    if is_mvt_valid(tt_new, mvt):
        v1 = is_lut_valid(tt_new, mvt)
        v2 = is_lut_valid(tt, mvt)
        v = set(v1).intersection(v2)
        if len(v) > 0:
            return list(v)
    return []

def is_superset_with_pos(a, b):
    i,j = 0,0
    pos = list()
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            pos.append(i)
            j += 1
        i += 1
    return (j == len(b)), pos

def proj_tt(inps1, inps2, pos2, tt2):
    proj = bool_val_combs[len(inps1)][:,pos2] @ powers_of_2[:len(inps2)][::-1]
    return tt2[proj]

class NodeMerger:
    def __init__(self):
        self.nodes_merged = dict() # inp_nodes[v] = [...] the list of nodes to be merged into v
        self.merged_into = dict() # node v is merged into u if merge_into[v] = u
        self.merged_lincomb = set()

    def merge(self, dst, src):
        assert(not self.is_merged(src))

        while dst in self.merged_into:
            dst = self.merged_into[dst]

        t = self.nodes_merged.setdefault(dst, [])
        t.append(src)
        if src in self.nodes_merged:
            t.extend(self.nodes_merged.pop(src))

        for n in t:
            self.merged_into[n] = dst

    def merge_lincomb(self, src):
        self.merged_lincomb.add(src)

    def is_merged(self, src):
        return src in self.merged_into or src in self.merged_lincomb

    def nb_merges(self):
        return sum(map(len, self.nodes_merged.values()))

    def map_nodes(self, nodes):
        return list(map(lambda v: self.merged_into.get(v, v), nodes))

    def __repr__(self):
        return f"{self.nb_merges()} - {self.nodes_merged}"



parser = argparse.ArgumentParser(
    description="Compute/estimate LBF circuits",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("input", help="input lbf filename")
parser.add_argument("output", nargs='?', default=None, help="output lbf filename")
parser.add_argument("--verbose", "-v", action="count", default=0)

# args = parser.parse_args(["adder_4_search.lbf"])
args = parser.parse_args()

levels = [logging.CRITICAL, logging.ERROR,
          logging.WARNING, logging.INFO, logging.DEBUG]
level = levels[min(args.verbose, len(levels) - 1)]
logging.basicConfig(level=level)

circuit = LutEnvParser.parse_file(args.input)
print(f"Input circuit: {circuit.stats()}")

node_succs = dict()
node_preds = dict()
out_ids = set(circuit.outputs.keys())

for node in circuit.instructions:
    node_succs[node.name] = list()
    node_preds[node.name] = list()

    match node:
        case LutExecEnv.LinearProd(name=name, coef_vals=coef_vals):
            coef_vals.sort(key=lambda e: e[1].name)
            inputs = list(map(lambda e: e[1].name, coef_vals))
            for inp in inputs:
                node_succs[inp].append(name)
                node_preds[name].append(inp)
        case LutExecEnv.Bootstrap(name=name, val=inp):
            node_succs[inp.name].append(name)
            node_preds[name].append(inp.name)

for out, val in circuit.outputs.items():
    node_preds[out] = list()
    node_preds[out].append(val.name)


merger = NodeMerger()
bootstrap_inps = dict()
lincombs = dict()

print("Find nodes used in more than 1 bootstrappings...")
for i, node in enumerate(circuit.instructions):
    if i % 1000 == 0:
        print(f"{i} / {len(circuit.instructions)}\r", end="")

    match node:
        case LutExecEnv.LinearProd(name=name, coef_vals=coef_vals, const_coef=const_coef):
            if name not in out_ids:
                coefs = list(map(lambda e: e[0], coef_vals))
                inps = list(map(lambda e: e[1].name, coef_vals))
                assert(inps == sorted(inps))
                assert(len(inps) == len(set(inps)))
                lincombs[name] = (inps, comp_mvt(coefs, const_coef))

        case LutExecEnv.Bootstrap(name=out, val=val, table=table, is_multi=is_multi):
            assert(not is_multi)

            tv, inp = np.array(table), val.name

            if len(lincombs[inp]) > 2: # lincomb already seen (ie already used in an FBS)
                merger.merge(lincombs[inp][2], out)
                continue

            for v in node_preds[inp]:
                bootstrap_inps.setdefault(v, set()).add(out)

            inps, mvt = lincombs[inp]
            lincombs[inp] = (inps, mvt, out, tv)

print(f"Merged nodes in the input circuit: {merger.nb_merges()}")

print("Find possible bootstrap merges...")
node_pos = dict(map(lambda e: (e[1].name, e[0]), enumerate(circuit.instructions)))

# check that u is not an ancestors of v and vice-versa
def check_not_ancestor(node_preds, node_pos, u, v):
    if node_pos[u] < node_pos[v]:
        u, v = v, u
    to_visit = set([u])
    while to_visit:
        w = to_visit.pop()
        if w == v: return False
        if node_pos[w] >= node_pos[v]:
            to_visit.update(node_preds[w])
    return True


visited = set()
for i, us in enumerate(bootstrap_inps.values()):
    if i % 1000 == 0:
        print(f"{i} / {len(bootstrap_inps.values())}\r", end="")

    us = list(us)
    for k, out_i in enumerate(us):
        if merger.is_merged(out_i): continue

        assert(len(node_preds[out_i]) == 1)
        inp_i = node_preds[out_i][0]
        inps_i, mvt_i, _, tv_i = lincombs[inp_i]
        tt_i = tv_i[mvt_i]

        for out_j in us[k+1:]:
            if merger.is_merged(out_j): continue
            if (out_i, out_j) in visited or (out_j, out_i) in visited: continue
            visited.add((out_i, out_j))
            visited.add((out_j, out_i))

            assert(len(node_preds[out_j]) == 1)
            inp_j = node_preds[out_j][0]
            inps_j, mvt_j, _, tv_j = lincombs[inp_j]

            b, pos = is_superset_with_pos(inps_i, inps_j)

            if b and len(inps_i) == len(inps_j):
                tt_j = tv_j[mvt_j]

                if v := is_lut_valid_new_mvt(tt_i, tt_j, mvt_j) and check_not_ancestor(node_preds, node_pos, out_i, out_j):
                    merger.merge(out_j, out_i)
                    merger.merge_lincomb(inp_i)
                    break
                if v := is_lut_valid_new_mvt(tt_j, tt_i, mvt_i) and check_not_ancestor(node_preds, node_pos, out_i, out_j):
                    merger.merge(out_i, out_j)
                    merger.merge_lincomb(inp_j)
                continue

            if b:
                tt_j = tv_j[mvt_j]

                tt_j_p = proj_tt(inps_i, inps_j, pos, tt_j)
                if v := is_lut_valid_new_mvt(tt_j_p, tt_i, mvt_i) and check_not_ancestor(node_preds, node_pos, out_i, out_j):
                    merger.merge(out_i, out_j)
                    merger.merge_lincomb(inp_j)

            b, pos = is_superset_with_pos(inps_j, inps_i)
            if b:
                tt_j = tv_j[mvt_j]

                tt_i_p = proj_tt(inps_j, inps_i, pos, tt_i)
                if v := is_lut_valid_new_mvt(tt_i_p, tt_j, mvt_j) and check_not_ancestor(node_preds, node_pos, out_i, out_j):
                    merger.merge(out_j, out_i)
                    merger.merge_lincomb(inp_i)
                    break


print(f"Merged nodes: {merger.nb_merges()}")

# reductions.sort(key=lambda e: e[1])

bootstraps_map = dict()
for id_i, src_nodes in merger.nodes_merged.items():
    boot_i = circuit.instructions[node_pos[id_i]]
    outs = [boot_i.name]
    tables = [boot_i.table]

    lincomb_i = boot_i.val
    coefs_i = list(map(lambda e: e[0], lincomb_i.coef_vals))
    inps_i = list(map(lambda e: e[1].name, lincomb_i.coef_vals))
    mvt_i = comp_mvt(coefs_i, lincomb_i.const_coef)

    for id_j in src_nodes:
        assert(id_j != id_i)

        boot_j = circuit.instructions[node_pos[id_j]]

        lincomb_j = boot_j.val
        coefs_j = list(map(lambda e: e[0], lincomb_j.coef_vals))
        inps_j = list(map(lambda e: e[1].name, lincomb_j.coef_vals))
        mvt_j = comp_mvt(coefs_j, lincomb_j.const_coef)

        tv_j = boot_j.table
        tt_j = np.array(tv_j)[mvt_j]

        b, pos = is_superset_with_pos(inps_i, inps_j)
        assert(b)
        tt_j_p = proj_tt(inps_i, inps_j, pos, tt_j)

        #TODO: use same v for all merged nodes
        v = is_lut_valid_new_mvt(tt_j_p, tt_i, mvt_i)
        assert(len(v) > 0)

        tv_j_p = comp_fbs_test_vector(tt_j_p, mvt_i, v[0])

        outs.append(id_j)
        tables.append(tv_j_p)

    bootstraps_map[id_i] = (outs, tables)


# find number of remaining predecessors to visit for each node in the new circuit
to_visit = list() # nodes to visit
remaining_pred_to_visit = dict()
for node in circuit.instructions:
    u = node.name
    if merger.is_merged(u):
        continue

    remaining_pred_to_visit[u] = len(node_preds[u])

    match node:
        case LutExecEnv.Input(name=out):
            to_visit.append(out)
        case LutExecEnv.Const(name=out):
            to_visit.append(out)

for u in circuit.outputs.keys():
    remaining_pred_to_visit[u] = len(node_preds[u])

new_circuit = LutExecEnv()
node_val = dict()
visited = set() # visited nodes
while to_visit:
    u = to_visit.pop(0)
    if u in visited or merger.is_merged(u): continue

    # print(f"Next node {u}")
    visited.add(u)

    node = circuit.instructions[node_pos[u]]
    node_outputs = list()
    match node:
        case LutExecEnv.Input(name=out):
            # print(f"Input {out}")
            node_val[out] = new_circuit.input(out)
            node_outputs = [out]
        case LutExecEnv.Const(name=out, value=val):
            # print(f"Const {out} {val}")
            node_val[out] = new_circuit.const(val)
            node_outputs = [out]
        case LutExecEnv.LinearProd(name=out, coef_vals=coef_vals, const_coef=const_coef):
            inps = list(map(lambda e: e[1].name, coef_vals))
            coefs = list(map(lambda e: e[0], coef_vals))
            # print(f"lincombA {out} {inps} {coefs}")
            vals = list(map(lambda inp: node_val[inp], inps))
            node_val[out] = new_circuit.linear(coefs, vals, const_coef)
            # print(f"\tlincombB {node_val[out].name} {list(map(lambda e: e.name, vals))} {coefs}")
            node_outputs = [out]
        case LutExecEnv.BootstrapMulti():
            assert(False)
        case LutExecEnv.Bootstrap(name=out, val=val, table=table, is_multi=is_multi):
            assert(not is_multi)
            # print(f"Bootstrap {out} <- {val.name} {table} {is_multi}")
            val = node_val[val.name]
            if out in bootstraps_map:
                node_outputs, tables = bootstraps_map[out]
                # print(f"\tBootstrapMulti A {val.name} {node_outputs} {tables}")
                _, bootstraps = new_circuit.bootstrap_multi(val, tables)
                # print(f"\tBootstrapMulti B {list(map(lambda e: e.name, bootstraps))}")
                for out, bootstrap in zip(node_outputs, bootstraps):
                    node_val[out] = bootstrap
            else:
                node_val[out] = new_circuit.bootstrap(val, table)
                # print(f"\tBootstrapA {node_val[out].name} <- {val.name} {table} {is_multi}")
                node_outputs = [out]

    for u in node_outputs:
        for v in node_succs[u]:
            if merger.is_merged(v): continue
            assert(remaining_pred_to_visit[v] > 0)
            remaining_pred_to_visit[v] -= 1
            if remaining_pred_to_visit[v] == 0:
                to_visit.append(v)

for out, val in circuit.outputs.items():
    val = node_val[val.name]
    new_circuit.output(out, val)

print(f"Output circuit: {new_circuit.stats()}")


inp_ids = list(map(lambda inp: inp.name, filter(lambda node: isinstance(node, LutExecEnv.Input), circuit.instructions)))

np.random.seed(42)
input_vals = dict(
    map(lambda inp: (inp, np.random.randint(0, 2, (4))), inp_ids))
output_values1 = circuit.eval(input_vals)
output_values2 = new_circuit.eval(input_vals)

assert(output_values1.keys() == output_values2.keys())
for k in output_values1.keys():
    equal = np.all(output_values1[k] == output_values2[k])
    if not equal:
        print(f"output {k} do not match {output_values1[k]} {output_values2[k]}")
    assert(equal)

new_circuit.write_lbf(open(args.output, "w"))

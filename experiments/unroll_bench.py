#!/usr/bin/env python3
"""Unroll a sequential ISCAS BENCH circuit into a combinational BENCH circuit."""

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


UNARY_GATES = {"BUF", "BUFF", "INV", "NOT"}
NARY_GATES = {"AND", "NAND", "NOR", "OR", "XNOR", "XOR"}
CONSTANTS = {"FALSE", "TRUE"}


@dataclass(frozen=True)
class Gate:
    output: str
    operation: str
    inputs: tuple[str, ...]


@dataclass(frozen=True)
class Circuit:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    latches: tuple[Gate, ...]
    gates: tuple[Gate, ...]


def invocation(expression):
    expression = expression.strip()
    open_paren = expression.find("(")
    close_paren = expression.rfind(")")
    if open_paren <= 0 or close_paren != len(expression) - 1:
        raise ValueError(f"invalid invocation: {expression!r}")
    operation = expression[:open_paren].strip().upper()
    arguments = tuple(
        argument.strip()
        for argument in expression[open_paren + 1 : close_paren].split(",")
    )
    if not arguments or any(not argument for argument in arguments):
        raise ValueError(f"invalid invocation: {expression!r}")
    return operation, arguments


def logical_lines(source):
    current = ""
    for raw_line in source.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        continued = line.endswith("\\")
        current += (" " if current else "") + line.removesuffix("\\").strip()
        if not continued:
            yield current
            current = ""
    if current:
        yield current


def parse_bench(source):
    inputs = []
    outputs = []
    latches = []
    gates = []
    defined = set(CONSTANTS)

    for line in logical_lines(source):
        if "=" not in line:
            declaration, names = invocation(line)
            if len(names) != 1 or declaration not in {"INPUT", "OUTPUT"}:
                raise ValueError(f"invalid declaration: {line!r}")
            name = names[0]
            if declaration == "INPUT":
                if name in defined:
                    raise ValueError(f"duplicate wire: {name!r}")
                inputs.append(name)
                defined.add(name)
            else:
                outputs.append(name)
            continue

        output, expression = (part.strip() for part in line.split("=", 1))
        if not output or output in defined:
            raise ValueError(f"duplicate or empty wire: {output!r}")
        operation, gate_inputs = invocation(expression)
        gate = Gate(output, operation, gate_inputs)
        if operation == "DFF":
            if len(gate_inputs) != 1:
                raise ValueError(f"DFF {output!r} requires one input")
            latches.append(gate)
        elif operation in UNARY_GATES:
            if len(gate_inputs) != 1:
                raise ValueError(f"{operation} {output!r} requires one input")
            gates.append(gate)
        elif operation in NARY_GATES:
            if len(gate_inputs) < 2:
                raise ValueError(f"{operation} {output!r} requires at least two inputs")
            gates.append(gate)
        else:
            raise ValueError(f"unsupported operation {operation!r} at {output!r}")
        defined.add(output)

    if not inputs and not latches:
        raise ValueError("circuit has no inputs or DFF state")
    if not outputs:
        raise ValueError("circuit has no outputs")
    unknown_outputs = set(outputs) - defined
    if unknown_outputs:
        raise ValueError(f"undefined outputs: {sorted(unknown_outputs)}")

    sources = set(inputs) | {latch.output for latch in latches} | CONSTANTS
    gate_outputs = {gate.output for gate in gates}
    unknown = {name for gate in gates for name in gate.inputs} - sources - gate_outputs
    if unknown:
        raise ValueError(f"cannot order gates: unknown wires {sorted(unknown)}")

    by_output = {gate.output: gate for gate in gates}
    dependencies = {
        gate.output: set(gate.inputs) & gate_outputs for gate in gates
    }
    dependents = defaultdict(list)
    for output, producers in dependencies.items():
        for producer in producers:
            dependents[producer].append(output)

    ready = deque(gate.output for gate in gates if not dependencies[gate.output])
    ordered = []
    while ready:
        output = ready.popleft()
        ordered.append(by_output[output])
        for dependent in dependents[output]:
            dependencies[dependent].remove(output)
            if not dependencies[dependent]:
                ready.append(dependent)
    if len(ordered) != len(gates):
        raise ValueError("cannot order gates: combinational cycle")

    unknown_latch_inputs = {
        latch.inputs[0] for latch in latches
    } - sources - gate_outputs
    if unknown_latch_inputs:
        raise ValueError(f"undefined DFF inputs: {sorted(unknown_latch_inputs)}")
    return Circuit(tuple(inputs), tuple(outputs), tuple(latches), tuple(ordered))


def framed(name, frame):
    return name if name in CONSTANTS else f"{name}__f{frame}"


def unroll(circuit, frames):
    if frames < 1:
        raise ValueError("frames must be positive")

    lines = [f"# {frames}-frame unrolling; initial DFF state is supplied as inputs."]
    for frame in range(frames):
        lines.extend(f"INPUT({framed(name, frame)})" for name in circuit.inputs)
    lines.extend(f"INPUT({framed(latch.output, 0)})" for latch in circuit.latches)
    lines.append("")
    for frame in range(frames):
        lines.extend(f"OUTPUT({framed(name, frame)})" for name in circuit.outputs)

    for frame in range(frames):
        lines.append("")
        lines.append(f"# Frame {frame}")
        if frame > 0:
            for latch in circuit.latches:
                source = framed(latch.inputs[0], frame - 1)
                lines.append(f"{framed(latch.output, frame)} = BUFF({source})")
        for gate in circuit.gates:
            inputs = ", ".join(framed(name, frame) for name in gate.inputs)
            lines.append(f"{framed(gate.output, frame)} = {gate.operation}({inputs})")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="sequential ISCAS BENCH file")
    parser.add_argument("output", type=Path, help="combinational BENCH file")
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()

    try:
        circuit = parse_bench(args.input.read_text())
        result = unroll(circuit, args.frames)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    try:
        args.output.write_text(result)
    except OSError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

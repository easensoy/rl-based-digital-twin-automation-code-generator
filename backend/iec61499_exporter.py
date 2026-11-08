"""
Export a trained DQN policy as PLC code that executes the policy itself.

``PLCCodeGenerator`` (plc_generator.py) only copies epsilon and the step count
into a fixed template. This module instead turns the Q-network weights into:

* an IEC 61499 Basic FB type (``<NAME>.fbt``) in the XML dialect EcoStruxure
  Automation Expert (EAE) reads: events INIT/REQ -> INITO/CNF with WITH
  associations, an ECC START -> INIT/REQ -> START, and ST algorithms that run
  the forward pass, the argmax and the action-to-setpoint table;
* optionally the same logic as an IEC 61131-3 FUNCTION_BLOCK (``<NAME>.st``)
  for CODESYS / TwinCAT style tools.

The network must be the ``DQNNetwork`` of rl_agent.py, or any stack of
``nn.Linear`` layers applied in state_dict order with ReLU between them and
none after the last (dropout is an identity at inference). All arithmetic is
REAL (float32), as in PyTorch.

Weights are stored as internal REAL arrays with ``InitialValue="[...]"``,
EAE's own array-initialiser format. EAE 25.0's FBTypeCompiler.exe overflowed
its stack (0xC00000FD) on one InitialValue with 6,500 elements (6,000 were
fine), so each weight matrix is split by rows into arrays of at most
``chunk_elems`` (default 2,048) elements. The alternative, one assignment per
weight in the INIT algorithm, was rejected: the same compiler overflowed at
16,384 statements and needed about 147 bytes of code per assignment.

Usage (from the repo root):
    .venv\\Scripts\\python.exe scripts\\export_iec61499_fb.py ^
        visual_outputs\\run\\policy_snapshot_20261106_190932.pt --verify 2000
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib
import math
import sys
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import quoteattr

import numpy as np

# ---------------------------------------------------------------------------
# Process model, copied from backend/rl_agent.py
# ---------------------------------------------------------------------------
#: State vector layout, from the comment in AutomationEnvironment.reset().
STATE_NAMES: Tuple[str, ...] = (
    "POS_X", "POS_Y", "POS_Z", "VEL_X", "VEL_Y", "VEL_Z",
    "TEMP", "PRESSURE", "VIBRATION", "LOAD", "QUALITY", "ENERGY",
)

#: AutomationEnvironment._apply_action(): action -> (name, speed, pressure, temp).
ACTION_SETPOINTS: Tuple[Tuple[str, float, float, float], ...] = (
    ("Conservative", 0.8, 0.9, 0.85),
    ("Normal", 1.0, 1.0, 1.0),
    ("Aggressive", 1.2, 1.1, 1.1),
    ("Energy saving", 0.6, 0.8, 0.8),
    ("Quality focus", 1.1, 1.05, 0.95),
    ("Balanced", 0.9, 0.95, 0.9),
)
#: _apply_action() falls back to action 1 ("Normal") for unknown actions.
DEFAULT_ACTION = 1

SETPOINT_OUTPUTS = ("SPEED_SP", "PRESSURE_SP", "TEMP_SP")

#: Largest array InitialValue known to compile with EAE 25.0 FBTypeCompiler.exe.
EAE_LARGEST_TESTED_INITIAL_VALUE = 6000
DEFAULT_CHUNK_ELEMS = 2048

# Internal names used by the generated algorithm (interface names must differ).
_LOOP_VARS = ("I", "J", "K", "A")
_ACC = "ACC"
_INPUT_BUFFER = "X"
_Q = "Q"

# ECC layout copied from EAE's own Basic FB template (FBBasic.Template.xft).
_ECC_STATES = (
    ("START", "Initial State", "552.9412", "429.4117", None, None),
    ("INIT", "Initialization", "923.5294", "141.1765", "INIT", "INITO"),
    ("REQ", "Normal execution", "217.647", "752.9412", "REQ", "CNF"),
)
_ECC_TRANSITIONS = (
    ("START", "INIT", "INIT", "923.5294", "429.4117"),
    ("INIT", "START", "1", "552.9412", "141.1765"),
    ("START", "REQ", "REQ", "552.9412", "600.0"),
    ("REQ", "START", "1", "217.647", "429.4117"),
)


# ---------------------------------------------------------------------------
# Policy weights
# ---------------------------------------------------------------------------
@dataclass
class PolicyWeights:
    """Linear layers (W of shape (out, in), b of shape (out,)), float32."""

    layers: List[Tuple[np.ndarray, np.ndarray]]
    source: Dict[str, object] = field(default_factory=dict)

    @property
    def sizes(self) -> List[int]:
        return [self.layers[0][0].shape[1]] + [W.shape[0] for W, _ in self.layers]

    @property
    def n_inputs(self) -> int:
        return self.sizes[0]

    @property
    def n_actions(self) -> int:
        return self.sizes[-1]

    @property
    def n_params(self) -> int:
        return int(sum(W.size + b.size for W, b in self.layers))

    @property
    def macs(self) -> int:
        """Multiply-accumulates per forward pass (one per weight)."""
        return int(sum(W.size for W, _ in self.layers))

    @property
    def sha256(self) -> str:
        """sha256 of the float32 little-endian bytes of W1, b1, W2, b2, ..."""
        h = hashlib.sha256()
        for W, b in self.layers:
            h.update(np.ascontiguousarray(W, dtype="<f4").tobytes())
            h.update(np.ascontiguousarray(b, dtype="<f4").tobytes())
        return h.hexdigest()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Float64 reference forward pass (for diagnostics only)."""
        h = np.asarray(x, dtype=np.float64)
        for i, (W, b) in enumerate(self.layers):
            h = h @ W.astype(np.float64).T + b.astype(np.float64)
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return h


def _to_numpy(t) -> np.ndarray:
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.asarray(t)


def policy_from_state_dict(state_dict, source: Optional[Dict[str, object]] = None) -> PolicyWeights:
    """Build PolicyWeights from a DQNNetwork-style state_dict (tensors or arrays)."""
    keys = list(state_dict.keys())
    layers = []
    used = set()
    for k in keys:
        if not k.endswith(".weight"):
            continue
        prefix = k[: -len(".weight")]
        bk = prefix + ".bias"
        if bk not in state_dict:
            raise ValueError(f"{k} has no bias {bk}")
        W = _to_numpy(state_dict[k]).astype(np.float32)
        b = _to_numpy(state_dict[bk]).astype(np.float32)
        if W.ndim != 2 or b.shape != (W.shape[0],):
            raise ValueError(f"{prefix}: expected a Linear layer, got weight {W.shape}, bias {b.shape}")
        if not (np.all(np.isfinite(W)) and np.all(np.isfinite(b))):
            raise ValueError(f"{prefix}: non-finite weights")
        layers.append((W, b))
        used.update((k, bk))
    extra = [k for k in keys if k not in used]
    if extra:
        raise ValueError(f"only Linear layers are supported; unexpected entries {extra}")
    if not layers:
        raise ValueError("no Linear layers found")
    for (W0, _), (W1, _) in zip(layers, layers[1:]):
        if W1.shape[1] != W0.shape[0]:
            raise ValueError(f"layer shapes do not chain: {W0.shape} -> {W1.shape}")
    return PolicyWeights(layers, dict(source or {}))


def load_policy(path, unsafe: bool = False) -> PolicyWeights:
    """Load a .pt file written by visual_outputs/scripts/run_instrumented_server.py
    ({'state_dict': ...}), a DQNAgent.get_optimal_policy() dict ({'model_state': ...})
    or a bare state_dict. Loads with weights_only=True unless unsafe=True."""
    import torch

    path = Path(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # TypedStorage deprecation noise in torch 2.1
        obj = torch.load(path, map_location="cpu", weights_only=not unsafe)
    meta: Dict[str, object] = {"file": path.name, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
        for k in ("epsilon", "gradient_steps", "episodes_completed", "seed"):
            if k in obj:
                meta[k] = obj[k]
    elif isinstance(obj, dict) and "model_state" in obj:
        sd = obj["model_state"]
        for k in ("epsilon", "training_steps"):
            if k in obj:
                meta[k] = obj[k]
    else:
        sd = obj
    return policy_from_state_dict(sd, meta)


# ---------------------------------------------------------------------------
# REAL literals
# ---------------------------------------------------------------------------
def real_literal(value) -> str:
    """Shortest IEC 61131-3 REAL literal that reads back as exactly float32(value).

    Form: digits '.' digits [ 'E' [sign] digits ], e.g. 0.85, -0.012345678, 1.5E-5."""
    f = np.float32(value)
    if not np.isfinite(f):
        raise ValueError(f"cannot write {value!r} as a REAL literal")
    if f == 0:
        return "-0.0" if np.signbit(f) else "0.0"
    pos = np.format_float_positional(f, unique=True, trim="0")
    sci = np.format_float_scientific(f, unique=True, trim="0", exp_digits=1).replace("e", "E")
    return pos if len(pos) <= len(sci) else sci


def _iec_from_python_repr(s: str) -> str:
    mant, _, exp = s.lower().partition("e")
    if "." not in mant:
        mant += ".0"
    return mant + ("E" + exp if exp else "")


def real_literals(values) -> List[str]:
    """real_literal() for every element, checked to read back bit-exactly through
    decimal -> double -> float32 as well (a compiler may parse that way)."""
    vals = np.ascontiguousarray(np.asarray(values, dtype=np.float32).ravel())
    lits = [real_literal(v) for v in vals]
    back = np.array([float(s) for s in lits], dtype=np.float64).astype(np.float32)
    for i in np.nonzero(back.view(np.uint32) != vals.view(np.uint32))[0]:
        lit = _iec_from_python_repr(repr(float(vals[i])))  # 17 digits: always exact
        if np.float32(float(lit)).view(np.uint32) != vals[i].view(np.uint32):
            raise AssertionError(f"no exact REAL literal for {vals[i]!r}")
        lits[i] = lit
    return lits


# ---------------------------------------------------------------------------
# Weight arrays
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WeightChunk:
    """Rows [row_start, row_stop) of layer `layer`'s weight matrix, row-major."""

    layer: int
    index: int
    name: str
    row_start: int
    row_stop: int
    n_in: int

    @property
    def size(self) -> int:
        return (self.row_stop - self.row_start) * self.n_in


def plan_chunks(policy: PolicyWeights, chunk_elems: int = DEFAULT_CHUNK_ELEMS) -> List[List[WeightChunk]]:
    if chunk_elems < 1:
        raise ValueError("chunk_elems must be >= 1")
    if chunk_elems > EAE_LARGEST_TESTED_INITIAL_VALUE:
        warnings.warn(f"chunk_elems={chunk_elems}: EAE 25.0 FBTypeCompiler.exe was only seen to compile "
                      f"InitialValue arrays up to {EAE_LARGEST_TESTED_INITIAL_VALUE} elements")
    plan = []
    for layer, (W, b) in enumerate(policy.layers, start=1):
        n_out, n_in = W.shape
        if n_in > chunk_elems or n_out > chunk_elems:
            raise ValueError(f"layer {layer} ({n_in} -> {n_out}) needs rows or a bias longer than "
                             f"chunk_elems={chunk_elems}; raise chunk_elems")
        rows = chunk_elems // n_in
        n_chunks = math.ceil(n_out / rows)
        width = len(str(n_chunks - 1))
        plan.append([WeightChunk(layer, c, f"W{layer}_{c:0{width}d}", c * rows, min(n_out, (c + 1) * rows), n_in)
                     for c in range(n_chunks)])
    return plan


# ---------------------------------------------------------------------------
# Generated code
# ---------------------------------------------------------------------------
@dataclass
class ExportOptions:
    name: str = "DQN_POLICY"
    namespace: str = "Main"
    state_input: str = "scalar"        # 'scalar': 12 named REAL inputs; 'array': STATE : ARRAY[0..11] OF REAL
    action_output: str = "ACTION"
    chunk_elems: int = DEFAULT_CHUNK_ELEMS
    organization: str = "WMG"
    author: str = "iec61499_exporter.py"
    version: str = "1.0"
    date: Optional[str] = None         # ISO date; default today

    def resolved_date(self) -> str:
        return self.date or _dt.date.today().isoformat()


def _ident_ok(s: str) -> bool:
    """IEC 61131-3 identifier: ASCII letters, digits, single underscores, no leading digit."""
    return (bool(s) and s.isascii() and s.replace("_", "A").isalnum() and not s[0].isdigit()
            and "__" not in s and not s.endswith("_"))


def _comment(text: str) -> str:
    """An ST comment that cannot terminate early."""
    return "(* " + text.replace("(*", "( *").replace("*)", "* )") + " *)"


def _input_names(policy: PolicyWeights, opts: ExportOptions) -> List[str]:
    if opts.state_input == "scalar":
        if policy.n_inputs != len(STATE_NAMES):
            raise ValueError(f"scalar inputs need {len(STATE_NAMES)} states; this network has "
                             f"{policy.n_inputs} (use state_input='array')")
        return list(STATE_NAMES)
    if opts.state_input == "array":
        return ["STATE"]
    raise ValueError("state_input must be 'scalar' or 'array'")


def _internal_names(policy: PolicyWeights, plan) -> List[str]:
    hidden = [f"H{i}" for i in range(1, len(policy.layers))]
    biases = [f"B{i}" for i in range(1, len(policy.layers) + 1)]
    chunks = [c.name for layer in plan for c in layer]
    return [_INPUT_BUFFER, *hidden, _Q, _ACC, *_LOOP_VARS, *biases, *chunks]


def _validate(policy: PolicyWeights, opts: ExportOptions, plan) -> None:
    if policy.n_actions != len(ACTION_SETPOINTS):
        raise ValueError(f"network has {policy.n_actions} outputs; the setpoint table has "
                         f"{len(ACTION_SETPOINTS)} actions")
    names = [opts.name, opts.action_output, *SETPOINT_OUTPUTS, *_input_names(policy, opts)]
    for n in names + [opts.namespace.replace(".", "_")]:
        if not _ident_ok(n):
            raise ValueError(f"{n!r} is not a valid IEC identifier")
    interface = [opts.action_output, *SETPOINT_OUTPUTS, *_input_names(policy, opts)]
    clash = {n.upper() for n in interface} & {n.upper() for n in _internal_names(policy, plan)}
    if clash or len({n.upper() for n in interface}) != len(interface):
        raise ValueError(f"interface names clash with each other or with internal names: {sorted(clash)}")


def _header_lines(policy: PolicyWeights, opts: ExportOptions) -> List[str]:
    sizes = "-".join(str(s) for s in policy.sizes)
    src = policy.source
    origin = []
    if "file" in src or "origin" in src:
        origin.append(f"source {src.get('file') or src['origin']}")
    for k in ("gradient_steps", "training_steps", "episodes_completed", "seed"):
        if k in src:
            origin.append(f"{k.replace('_', ' ')} {src[k]}")
    if "epsilon" in src:
        origin.append(f"epsilon {float(src['epsilon']):.6g}")
    lines = [
        _comment(f"{opts.name}: greedy action of a trained DQN Q-network, generated by backend/iec61499_exporter.py "
                 f"on {opts.resolved_date()}. Do not edit: re-export from the policy snapshot instead."),
        _comment(f"Network {sizes} (Linear + ReLU, last layer linear), {policy.n_params:,} parameters, "
                 f"{policy.macs:,} multiply-accumulates per REQ, REAL (float32) arithmetic."),
        _comment(f"Weights sha256 {policy.sha256}"),
    ]
    if origin:
        lines.insert(2, _comment(", ".join(origin)))
    return lines


def req_algorithm(policy: PolicyWeights, plan, opts: ExportOptions, header: bool = True) -> str:
    """ST body: latch the state, forward pass, argmax, setpoint table."""
    act = opts.action_output
    n_layers = len(policy.layers)
    L: List[str] = _header_lines(policy, opts) if header else []
    L.append("")
    L.append(_comment("1. State vector (rl_agent.AutomationEnvironment, values in [-2, 2] during training)"))
    if opts.state_input == "scalar":
        L += [f"{_INPUT_BUFFER}[{j}] := {nm};" for j, nm in enumerate(STATE_NAMES)]
    else:
        L += [f"FOR J := 0 TO {policy.n_inputs - 1} DO",
              f"    {_INPUT_BUFFER}[J] := STATE[J];",
              "END_FOR;"]
    for layer, ((W, _), chunks) in enumerate(zip(policy.layers, plan), start=1):
        n_out, n_in = W.shape
        last = layer == n_layers
        src = _INPUT_BUFFER if layer == 1 else f"H{layer - 1}"
        dst = _Q if last else f"H{layer}"
        kind = "linear, Q-values" if last else "ReLU"
        L.append("")
        L.append(_comment(f"{layer + 1}. Layer {layer} (fc{layer}): {n_in} -> {n_out}, {kind}; "
                          f"{dst}[i] = {'' if last else 'max(0, '}sum_j W[i, j] * {src}[j] + B{layer}[i]"
                          f"{'' if last else ')'}"))
        # The bias is added after the dot product, as torch.addmm does. On 12,000
        # rollout states this gave max |dQ| 5.5e-4 against PyTorch, against 6.7e-4
        # when the sum starts at the bias.
        result = f"{_ACC} + B{layer}[I]" if last else f"MAX({_ACC} + B{layer}[I], 0.0)"
        for ch in chunks:
            L += [
                _comment(f"rows {ch.row_start}..{ch.row_stop - 1} of fc{layer}.weight in {ch.name}, "
                         f"row-major: {ch.name}[(i - {ch.row_start}) * {n_in} + j]"),
                "K := 0;",
                f"FOR I := {ch.row_start} TO {ch.row_stop - 1} DO",
                f"    {_ACC} := 0.0;",
                f"    FOR J := 0 TO {n_in - 1} DO",
                f"        {_ACC} := {_ACC} + {ch.name}[K] * {src}[J];",
                "        K := K + 1;",
                "    END_FOR;",
                f"    {dst}[I] := {result};",
                "END_FOR;",
            ]
    L.append("")
    L.append(_comment(f"{n_layers + 2}. Greedy action: first index of the largest Q, "
                      f"the same tie-break as torch.argmax"))
    L += [f"{act} := 0;",
          f"FOR A := 1 TO {policy.n_actions - 1} DO",
          f"    IF {_Q}[A] > {_Q}[{act}] THEN",
          f"        {act} := A;",
          "    END_IF;",
          "END_FOR;",
          ""]
    L.append(_comment(f"{n_layers + 3}. Setpoints of the chosen action: AutomationEnvironment._apply_action "
                      f"(backend/rl_agent.py)"))
    L.append(f"CASE {act} OF")
    for a, (label, speed, pressure, temp) in enumerate(ACTION_SETPOINTS):
        L.append(f"    {a}: {_comment(label)}")
        for out, v in zip(SETPOINT_OUTPUTS, (speed, pressure, temp)):
            L.append(f"        {out} := {real_literal(v)};")
    _, speed, pressure, temp = ACTION_SETPOINTS[DEFAULT_ACTION]
    L.append("ELSE")
    L.append(f"    {_comment('not reachable (argmax is 0..5); same fallback as _apply_action: Normal')}")
    for out, v in zip(SETPOINT_OUTPUTS, (speed, pressure, temp)):
        L.append(f"    {out} := {real_literal(v)};")
    L.append("END_CASE;")
    return "\n".join(L) + "\n"


def init_algorithm(opts: ExportOptions) -> str:
    _, speed, pressure, temp = ACTION_SETPOINTS[DEFAULT_ACTION]
    lines = [_comment(f"Outputs start at action {DEFAULT_ACTION} (Normal), the fallback of "
                      f"AutomationEnvironment._apply_action. The weights are InitialValues: nothing to load."),
             f"{opts.action_output} := {DEFAULT_ACTION};"]
    lines += [f"{out} := {real_literal(v)};" for out, v in zip(SETPOINT_OUTPUTS, (speed, pressure, temp))]
    return "\n".join(lines) + "\n"


def _cdata(text: str) -> str:
    return "<![CDATA[" + text.replace("]]>", "]]]]><![CDATA[>") + "]]>"


def _attrs(*pairs: Tuple[str, Optional[str]]) -> str:
    return " ".join(f"{k}={quoteattr(str(v))}" for k, v in pairs if v is not None)


def fb_guid(opts: ExportOptions) -> str:
    """Stable GUID per type name, so a re-export replaces the same EAE type."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"urn:wmg-rl-digital-twin:fbtype:{opts.namespace}.{opts.name}"))


def _interface_decls(policy: PolicyWeights, opts: ExportOptions):
    """(name, type, array size, initial value, comment) for inputs and outputs."""
    if opts.state_input == "scalar":
        inputs = [(nm, "REAL", None, None, f"State[{j}] {nm.lower()} of rl_agent.AutomationEnvironment, "
                   f"trained range [-2, 2]") for j, nm in enumerate(STATE_NAMES)]
    else:
        inputs = [("STATE", "REAL", policy.n_inputs, None,
                   "State vector [" + ", ".join(n.lower() for n in STATE_NAMES) + "]")]
    actions = ", ".join(f"{a} {lbl}" for a, (lbl, *_rest) in enumerate(ACTION_SETPOINTS))
    _, speed, pressure, temp = ACTION_SETPOINTS[DEFAULT_ACTION]
    outputs = [(opts.action_output, "INT", None, str(DEFAULT_ACTION), f"Greedy action: {actions}"),
               ("SPEED_SP", "REAL", None, real_literal(speed), "Speed multiplier of the chosen action"),
               ("PRESSURE_SP", "REAL", None, real_literal(pressure), "Pressure multiplier of the chosen action"),
               ("TEMP_SP", "REAL", None, real_literal(temp), "Temperature multiplier of the chosen action")]
    return inputs, outputs


def _internal_decls(policy: PolicyWeights, plan, lits_by_name: Dict[str, List[str]]):
    """(name, type, array size, initial value list, comment) for internal variables."""
    n_layers = len(policy.layers)
    out = [(_INPUT_BUFFER, "REAL", policy.n_inputs, None, "State vector latched at REQ")]
    for i in range(1, n_layers):
        out.append((f"H{i}", "REAL", policy.layers[i - 1][0].shape[0], None, f"Layer {i} activations"))
    out.append((_Q, "REAL", policy.n_actions, None, "Q-value of every action (watch online to see the margin)"))
    out.append((_ACC, "REAL", None, None, "Dot-product accumulator"))
    out += [("I", "DINT", None, None, "Row index"), ("J", "DINT", None, None, "Column index"),
            ("K", "DINT", None, None, "Flat weight index within a weight array"),
            ("A", "INT", None, None, "Action index")]
    for layer, ((W, b), chunks) in enumerate(zip(policy.layers, plan), start=1):
        out.append((f"B{layer}", "REAL", b.size, lits_by_name[f"B{layer}"], f"fc{layer}.bias"))
        for ch in chunks:
            out.append((ch.name, "REAL", ch.size, lits_by_name[ch.name],
                        f"fc{layer}.weight rows {ch.row_start}..{ch.row_stop - 1}, row-major ({W.shape[1]} per row)"))
    return out


def _literal_table(policy: PolicyWeights, plan) -> Dict[str, List[str]]:
    table: Dict[str, List[str]] = {}
    for layer, ((W, b), chunks) in enumerate(zip(policy.layers, plan), start=1):
        table[f"B{layer}"] = real_literals(b)
        for ch in chunks:
            table[ch.name] = real_literals(W[ch.row_start:ch.row_stop, :])
    return table


def build_fbt(policy: PolicyWeights, opts: Optional[ExportOptions] = None,
              _lits: Optional[Dict[str, List[str]]] = None) -> str:
    """IEC 61499-2 Basic FB type, in the form EAE writes (.fbt)."""
    opts = opts or ExportOptions()
    plan = plan_chunks(policy, opts.chunk_elems)
    _validate(policy, opts, plan)
    lits = _lits or _literal_table(policy, plan)
    inputs, outputs = _interface_decls(policy, opts)
    in_names = [d[0] for d in inputs]
    out_names = [d[0] for d in outputs]
    sizes = "-".join(str(s) for s in policy.sizes)

    def var(d, indent):
        name, typ, size, init, comment = d
        if isinstance(init, list):
            init = "[" + ",".join(init) + "]"
        a = _attrs(("Name", name), ("Type", typ), ("ArraySize", size), ("InitialValue", init), ("Comment", comment))
        return f"{indent}<VarDeclaration {a} />"

    def event(name, comment, withs, indent):
        if not withs:
            return [f'{indent}<Event {_attrs(("Name", name), ("Comment", comment))} />']
        return ([f'{indent}<Event {_attrs(("Name", name), ("Comment", comment))}>']
                + [f'{indent}  <With {_attrs(("Var", w))} />' for w in withs]
                + [f"{indent}</Event>"])

    fb_comment = f"Greedy DQN policy {sizes}: forward pass, argmax and setpoint table"
    source = policy.source.get("file") or policy.source.get("origin") or "a state_dict"
    remarks = f"Generated from {source}, weights sha256 {policy.sha256[:16]}"
    fbtype = _attrs(("GUID", fb_guid(opts)), ("Name", opts.name), ("Comment", fb_comment),
                    ("Namespace", opts.namespace))
    ident = _attrs(("Standard", "61499-2"), ("Classification", "Reinforcement learning"),
                   ("Function", "Greedy DQN policy"))
    version = _attrs(("Organization", opts.organization), ("Version", opts.version), ("Author", opts.author),
                     ("Date", opts.resolved_date()), ("Remarks", remarks))
    x = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<!DOCTYPE FBType SYSTEM "../LibraryElement.dtd">',
         f"<FBType {fbtype}>",
         f"  <Identification {ident} />",
         f"  <VersionInfo {version} />",
         "  <CompilerInfo />",
         "  <InterfaceList>",
         "    <EventInputs>"]
    x += event("INIT", "Initialization Request", [], "      ")
    x += event("REQ", "Normal Execution Request: compute the greedy action for the latched state", in_names, "      ")
    x += ["    </EventInputs>", "    <EventOutputs>"]
    x += event("INITO", "Initialization Confirm", out_names, "      ")
    x += event("CNF", "Execution Confirmation: action and setpoints are valid", out_names, "      ")
    x += ["    </EventOutputs>", "    <InputVars>"]
    x += [var(d, "      ") for d in inputs]
    x += ["    </InputVars>", "    <OutputVars>"]
    x += [var(d, "      ") for d in outputs]
    x += ["    </OutputVars>", "  </InterfaceList>", "  <BasicFB>", "    <InternalVars>"]
    x += [var(d, "      ") for d in _internal_decls(policy, plan, lits)]
    x += ["    </InternalVars>", "    <ECC>"]
    for name, comment, sx, sy, alg, out in _ECC_STATES:
        if alg is None:
            x.append(f'      <ECState {_attrs(("Name", name), ("Comment", comment), ("x", sx), ("y", sy))} />')
        else:
            x += [f'      <ECState {_attrs(("Name", name), ("Comment", comment), ("x", sx), ("y", sy))}>',
                  f'        <ECAction {_attrs(("Algorithm", alg), ("Output", out))} />',
                  "      </ECState>"]
    for src, dst, cond, tx, ty in _ECC_TRANSITIONS:
        a = _attrs(("Source", src), ("Destination", dst), ("Condition", cond), ("x", tx), ("y", ty))
        x.append(f"      <ECTransition {a} />")
    x += ["    </ECC>",
          f'    <Algorithm {_attrs(("Name", "INIT"), ("Comment", "Initialization algorithm: outputs to the Normal action"))}>',
          f"      <ST>{_cdata(init_algorithm(opts))}</ST>",
          "    </Algorithm>",
          f'    <Algorithm {_attrs(("Name", "REQ"), ("Comment", "Forward pass, argmax and setpoint table"))}>',
          f"      <ST>{_cdata(req_algorithm(policy, plan, opts))}</ST>",
          "    </Algorithm>",
          "  </BasicFB>",
          "</FBType>"]
    return "\n".join(x) + "\n"


def build_st(policy: PolicyWeights, opts: Optional[ExportOptions] = None,
             _lits: Optional[Dict[str, List[str]]] = None, per_line: int = 8) -> str:
    """The same logic as one IEC 61131-3 FUNCTION_BLOCK (outputs start at Normal)."""
    opts = opts or ExportOptions()
    plan = plan_chunks(policy, opts.chunk_elems)
    _validate(policy, opts, plan)
    lits = _lits or _literal_table(policy, plan)
    inputs, outputs = _interface_decls(policy, opts)

    def decl(d):
        name, typ, size, init, comment = d
        t = f"ARRAY[0..{size - 1}] OF {typ}" if size is not None else typ
        if isinstance(init, list):
            rows = [", ".join(init[i:i + per_line]) for i in range(0, len(init), per_line)]
            body = ",\n        ".join(rows)
            return [f"    {name} : {t} := [ {_comment(comment)}", f"        {body}", "    ];"]
        tail = f" := {init}" if init is not None else ""
        return [f"    {name} : {t}{tail}; {_comment(comment)}"]

    internal = _internal_decls(policy, plan, lits)
    L = _header_lines(policy, opts)
    L += [_comment("IEC 61131-3 form of the IEC 61499 Basic FB in the .fbt next to this file: call it with "
                   "the state, read the action and setpoints. Outputs start at the Normal action."),
          f"FUNCTION_BLOCK {opts.name}", "VAR_INPUT"]
    for d in inputs:
        L += decl(d)
    L += ["END_VAR", "VAR_OUTPUT"]
    for d in outputs:
        L += decl(d)
    L += ["END_VAR", "VAR"]
    for d in internal:
        if d[3] is None:
            L += decl(d)
    L += ["END_VAR", "VAR CONSTANT"]
    for d in internal:
        if d[3] is not None:
            L += decl(d)
    L += ["END_VAR", ""]
    L += req_algorithm(policy, plan, opts, header=False).rstrip("\n").split("\n")
    L += ["", "END_FUNCTION_BLOCK"]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
@dataclass
class ExportResult:
    fbt_path: Path
    st_path: Optional[Path]
    policy: PolicyWeights
    options: ExportOptions
    chunks: List[List[WeightChunk]]

    @property
    def fbt_bytes(self) -> int:
        return self.fbt_path.stat().st_size

    @property
    def st_bytes(self) -> Optional[int]:
        return self.st_path.stat().st_size if self.st_path else None

    @property
    def weight_bytes(self) -> int:
        return 4 * self.policy.n_params

    @property
    def instance_data_bytes(self) -> int:
        """Data of one FB instance (REAL 4 B, DINT 4 B, INT 2 B), before any runtime overhead."""
        p = self.policy
        activations = p.n_inputs + sum(W.shape[0] for W, _ in p.layers)  # X, H1..H{n-1}, Q
        io = 4 * p.n_inputs + 2 + 4 * len(SETPOINT_OUTPUTS)
        return self.weight_bytes + 4 * activations + 4 + 3 * 4 + 2 + io

    def summary(self) -> str:
        p = self.policy
        n_arrays = sum(len(c) for c in self.chunks)
        lines = [
            f"FB type        {self.options.namespace}.{self.options.name} (GUID {fb_guid(self.options)})",
            f"network        {'-'.join(map(str, p.sizes))}, {p.n_params:,} parameters, weights sha256 {p.sha256[:16]}...",
            f"per decision   {p.macs:,} multiply-accumulates, {sum(b.size for _, b in p.layers)} bias adds, "
            f"{sum(W.shape[0] for W, _ in p.layers[:-1])} ReLU (MAX), {p.n_actions - 1} comparisons",
            f"weight data    {self.weight_bytes:,} B as REAL in {n_arrays} weight + {len(p.layers)} bias arrays "
            f"(largest {max(c.size for layer in self.chunks for c in layer):,} elements)",
            f"instance data  ~{self.instance_data_bytes:,} B",
            f"{self.fbt_path.name:<14} {self.fbt_bytes:,} B  ({self.fbt_path})",
        ]
        if self.st_path:
            lines.append(f"{self.st_path.name:<14} {self.st_bytes:,} B  ({self.st_path})")
        return "\n".join(lines)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))  # LF line ends, no BOM


def export(policy: PolicyWeights, out_dir, opts: Optional[ExportOptions] = None, st: bool = True) -> ExportResult:
    opts = opts or ExportOptions()
    plan = plan_chunks(policy, opts.chunk_elems)
    _validate(policy, opts, plan)
    lits = _literal_table(policy, plan)
    out_dir = Path(out_dir)
    fbt_path = out_dir / f"{opts.name}.fbt"
    _write(fbt_path, build_fbt(policy, opts, _lits=lits))
    st_path = None
    if st:
        st_path = out_dir / f"{opts.name}.st"
        _write(st_path, build_st(policy, opts, _lits=lits))
    return ExportResult(fbt_path, st_path, policy, opts, plan)


# ---------------------------------------------------------------------------
# Optional verification (used by the CLI)
# ---------------------------------------------------------------------------
def _sibling(name: str):
    """Import a backend module whether this file was imported as backend.x or flat."""
    if __package__:
        return importlib.import_module(f"{__package__}.{name}")
    return importlib.import_module(name)


def rollout_states(n: int, seed: int = 0, env_config: Optional[dict] = None) -> np.ndarray:
    """n states visited by rl_agent.AutomationEnvironment under uniform random actions."""
    rl_agent = _sibling("rl_agent")
    saved = np.random.get_state()
    try:
        np.random.seed(seed)  # the environment draws from numpy's global RNG
        env = rl_agent.AutomationEnvironment(env_config or {})
        rng = np.random.default_rng(seed)
        out = []
        while len(out) < n:
            s, done = env.reset(), False
            while not done and len(out) < n:
                out.append(s)
                s, _, done = env.step(int(rng.integers(len(ACTION_SETPOINTS))))
        return np.array(out, dtype=np.float64)
    finally:
        np.random.set_state(saved)


def reference_network(policy: PolicyWeights):
    """rl_agent.DQNNetwork holding the policy's weights, in eval mode (dropout off).

    Building a network draws from torch's global RNG, which the training loop also
    uses; the RNG state is restored so a seeded training run is not disturbed."""
    import torch

    rl_agent = _sibling("rl_agent")
    if len(policy.layers) != 4 or len({W.shape[0] for W, _ in policy.layers[:-1]}) != 1:
        raise ValueError("the reference is rl_agent.DQNNetwork: 4 Linear layers with equal hidden size")
    saved = torch.get_rng_state()
    try:
        net = rl_agent.DQNNetwork(policy.n_inputs, policy.n_actions, hidden_size=policy.layers[0][0].shape[0])
    finally:
        torch.set_rng_state(saved)
    with torch.no_grad():
        for mod, (W, b) in zip((net.fc1, net.fc2, net.fc3, net.fc4), policy.layers):
            mod.weight.copy_(torch.from_numpy(W))
            mod.bias.copy_(torch.from_numpy(b))
    return net.eval()


def verify_fbt(fbt_path, policy: PolicyWeights, states, net=None, q_tolerance: float = 1e-3,
               tie_tolerance: float = 0.0, action_output: str = "ACTION") -> Dict[str, object]:
    """Run the .fbt (INIT, then REQ for every state) in st_interpreter and compare
    with DQNNetwork.eval() from rl_agent.py (``net``, or reference_network(policy)).

    ``ok`` requires all three ``checks``:

    * ``q_values``: max |Q difference| < q_tolerance;
    * ``argmax``: the FB's action output is the network's argmax on every state.
      A state where they differ is excused only when ``tie_tolerance`` > 0 and the
      FB's action is within tie_tolerance of the network's best Q-value (so the
      network's top two are that close too). Excused states are counted in
      ``near_tie_disagreements``; with the default 0.0 none is excused;
    * ``setpoints``: SPEED_SP, PRESSURE_SP and TEMP_SP equal the ACTION_SETPOINTS
      entry of the network's action (of the FB's own action on an excused state).

    ``examples`` lists up to five failing states. An FB that does not emit exactly
    one CNF with all four outputs fails every check, with the reason in ``error``."""
    import torch

    st_interpreter = _sibling("st_interpreter")
    net = net if net is not None else reference_network(policy)
    x32 = np.asarray(states, dtype=np.float32)
    n = len(x32)
    with torch.no_grad():
        q_ref = net(torch.from_numpy(x32)).numpy().astype(np.float64)
    report: Dict[str, object] = {
        "ok": False, "checks": {"q_values": False, "argmax": False, "setpoints": False},
        "states": n, "q_tolerance": q_tolerance, "tie_tolerance": tie_tolerance,
        "q_range": [float(q_ref.min()), float(q_ref.max())],
    }
    fb = st_interpreter.BasicFB(Path(fbt_path), n=n)
    fb.fire("INIT")
    decl_inputs = [nm for nm, d in fb.decls.items() if d.kind == "input"]
    if decl_inputs == ["STATE"]:
        inputs = {"STATE": x32.T}
    else:
        inputs = {nm: x32[:, j] for j, nm in enumerate(STATE_NAMES)}
    emitted = fb.fire("REQ", inputs)
    events = [e for e, _ in emitted]
    missing = [] if events != ["CNF"] else \
        [nm for nm in (action_output, *SETPOINT_OUTPUTS) if nm not in emitted[0][1]]
    if events != ["CNF"] or missing:
        report["error"] = (f"REQ emitted {events}, expected ['CNF']" if events != ["CNF"] else
                           f"CNF does not carry {', '.join(missing)} (WITH associations)")
        return report
    sample = emitted[0][1]
    q_fb = fb.get(_Q).T.astype(np.float64)
    a_fb = np.asarray(st_interpreter.lanes(sample[action_output], n)).astype(np.int64)
    sp_fb = np.stack([st_interpreter.lanes(sample[nm], n) for nm in SETPOINT_OUTPUTS], axis=1).astype(np.float32)

    a_ref = q_ref.argmax(axis=1)  # torch.argmax: the first maximum
    rows = np.arange(n)
    dq = np.abs(q_fb - q_ref)
    max_diff = float(dq.max())
    valid = (a_fb >= 0) & (a_fb < policy.n_actions)
    disagree = a_fb != a_ref
    shortfall = q_ref[rows, a_ref] - q_ref[rows, np.where(valid, a_fb, a_ref)]  # how much worse the FB's pick is
    near_tie = disagree & valid & (tie_tolerance > 0) & (shortfall <= tie_tolerance)
    table = np.array([row[1:] for row in ACTION_SETPOINTS], dtype=np.float32)
    sp_bad = np.any(sp_fb != table[np.where(near_tie, a_fb, a_ref)], axis=1)
    checks = {"q_values": max_diff < q_tolerance, "argmax": not bool(np.any(disagree & ~near_tie)),
              "setpoints": not bool(np.any(sp_bad))}

    failing = np.flatnonzero((dq.max(axis=1) >= q_tolerance) | (disagree & ~near_tie) | sp_bad)[:5]
    top2 = np.sort(q_ref, axis=1)[:, -2:]
    report.update({
        "ok": all(checks.values()),
        "checks": checks,
        "max_abs_q_diff": max_diff,
        "argmax_agreement": float(np.mean(~disagree)),
        "argmax_disagreements": int(np.sum(disagree)),
        "near_tie_disagreements": int(np.sum(near_tie)),
        "setpoint_mismatches": int(np.sum(sp_bad)),
        "action_histogram": np.bincount(a_fb[valid], minlength=policy.n_actions).tolist(),
        "invalid_actions": int(np.sum(~valid)),
        "examples": [{"state": int(i), "network_action": int(a_ref[i]), "fb_action": int(a_fb[i]),
                      "network_top2_gap": float(top2[i, 1] - top2[i, 0]), "max_abs_q_diff": float(dq[i].max()),
                      "fb_setpoints": [float(v) for v in sp_fb[i]],
                      "table_setpoints": [float(v) for v in table[a_ref[i]]]} for i in failing],
    })
    return report


def verification_summary(report: Dict[str, object]) -> str:
    """One line for logs and the CLI."""
    if "max_abs_q_diff" not in report:
        return f"{report['states']:,} states: FAILED, {report.get('error')}"
    failed = [name for name, passed in report["checks"].items() if not passed]
    ties = (f", {report['near_tie_disagreements']} near-tie disagreements excused "
            f"(tie tolerance {report['tie_tolerance']:g})" if report["near_tie_disagreements"] else "")
    return (f"{report['states']:,} states: argmax agreement {report['argmax_agreement']:.2%}{ties}, "
            f"max |dQ| {report['max_abs_q_diff']:.2e} (limit {report['q_tolerance']:g}), "
            f"{report['setpoint_mismatches']} setpoint mismatches -> "
            f"{'OK' if report['ok'] else 'FAILED: ' + ', '.join(failed)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Export a trained DQN policy as an IEC 61499 Basic FB (.fbt, EAE) and an "
                    "IEC 61131-3 FUNCTION_BLOCK (.st) that execute the network itself.")
    ap.add_argument("policy", help="policy .pt: {'state_dict': ...} snapshot, get_optimal_policy() dict or a state_dict")
    ap.add_argument("--out-dir", default="data/generated_code/iec61499", help="output folder (default: %(default)s)")
    ap.add_argument("--name", default="DQN_POLICY", help="FB type name, also the file name (default: %(default)s)")
    ap.add_argument("--namespace", default="Main", help="EAE namespace (default: %(default)s)")
    ap.add_argument("--state-input", choices=("scalar", "array"), default="scalar",
                    help="12 named REAL inputs (default) or one STATE : ARRAY[0..11] OF REAL")
    ap.add_argument("--action-name", default="ACTION", help="name of the INT action output (default: %(default)s)")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_ELEMS,
                    help="max elements per weight array (default: %(default)s; EAE 25.0 failed at 6,500)")
    ap.add_argument("--no-st", action="store_true", help="skip the IEC 61131-3 .st file")
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="run N AutomationEnvironment rollout states through the .fbt and compare with PyTorch")
    ap.add_argument("--eae-check", action="store_true",
                    help="compile the .fbt with EAE's FBTypeCompiler.exe, if installed")
    ap.add_argument("--unsafe-load", action="store_true", help="torch.load without weights_only (trusted files only)")
    args = ap.parse_args(argv)

    policy = load_policy(args.policy, unsafe=args.unsafe_load)
    opts = ExportOptions(name=args.name, namespace=args.namespace, state_input=args.state_input,
                         action_output=args.action_name, chunk_elems=args.chunk_size)
    result = export(policy, args.out_dir, opts, st=not args.no_st)
    print(result.summary())
    status = 0
    if args.verify:
        rep = verify_fbt(result.fbt_path, policy, rollout_states(args.verify, seed=0),
                         action_output=opts.action_output)
        print(f"verify         rollout {verification_summary(rep)}")
        if "action_histogram" in rep:
            print(f"               Q in [{rep['q_range'][0]:.1f}, {rep['q_range'][1]:.1f}], "
                  f"FB actions {rep['action_histogram']}")
        status |= 0 if rep["ok"] else 1
    if args.eae_check:
        eae = _sibling("eae_fbtype_compiler")
        exe = eae.find_fbtype_compiler()
        if exe is None:
            print("eae-check      FBTypeCompiler.exe not found (set EAE_FBTYPECOMPILER)")
            status |= 2
        else:
            rep = eae.compile_fbt(result.fbt_path, Path(args.out_dir) / "eae_build", exe)
            print(f"eae-check      {rep.summary()}")
            status |= 0 if rep.ok else 1
    return status


if __name__ == "__main__":
    sys.exit(main())

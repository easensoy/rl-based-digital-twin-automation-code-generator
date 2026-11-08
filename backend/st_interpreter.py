"""
Reference interpreter for the IEC 61131-3 Structured Text that
``iec61499_exporter.py`` generates, and for the IEC 61499 Basic FB execution
model around it.

Purpose: verify generated PLC code against the PyTorch network by executing
the generated text itself (the ``.fbt`` XML or the ``.st`` POU), not a
hand-written re-implementation of it.

Semantics
---------
* IEC ``REAL`` is IEEE-754 binary32 and ``LREAL`` is binary64. Every multiply
  and add is rounded to the declared type (numpy elementwise ops, no fused
  multiply-add), as on a PLC runtime.
* Typing is strict IEC 61131-3: no implicit INT<->REAL conversion, no
  narrowing assignment, array indices are bounds-checked, integer assignments
  are range-checked, inputs and constants are read-only, undeclared names are
  errors. Unsupported syntax raises ``STError``; nothing is skipped silently.
* IEC 61499 Basic FB: an input event latches only the data inputs WITH-
  associated to it, then the ECC is evaluated (an event-qualified transition
  can fire only on the first evaluation after the event arrives; guard-only
  transitions such as ``1`` keep firing). Entering a state runs its EC actions
  in order. An output event samples the outputs WITH-associated to it.

Execution is vectorised over N independent FB instances ("lanes"), so 10,000+
states go through a 35k-MAC algorithm in a few seconds. Control flow must be
the same in every lane, except IF and CASE, which run under a lane mask.

Supported subset: VAR / VAR_INPUT / VAR_OUTPUT / VAR CONSTANT declarations of
BOOL, SINT, INT, DINT, LINT, REAL, LREAL and 1-D arrays of them (with ``[..]``
and ``n(v)`` initialisers); assignment, IF/ELSIF/ELSE, CASE with integer
labels, FOR ... BY; operators ``+ - * / MOD``, comparisons, ``AND OR XOR NOT``,
unary minus; functions MAX, MIN, ABS. Comments ``(* *)`` and ``//``.

Only numpy is required.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "STError",
    "VarDecl",
    "BasicFB",
    "STFunctionBlock",
    "parse_array_size",
    "parse_initializer",
    "lanes",
]


class STError(Exception):
    """Raised for syntax, typing or runtime errors in the interpreted code."""


# --------------------------------------------------------------------------
# Types
# --------------------------------------------------------------------------
INT_RANGE = {
    "SINT": (-(2 ** 7), 2 ** 7 - 1),
    "INT": (-(2 ** 15), 2 ** 15 - 1),
    "DINT": (-(2 ** 31), 2 ** 31 - 1),
    "LINT": (-(2 ** 63), 2 ** 63 - 1),
}
INT_RANK = {"SINT": 1, "INT": 2, "DINT": 3, "LINT": 4}
REAL_DTYPE = {"REAL": np.float32, "LREAL": np.float64}
REAL_RANK = {"REAL": 1, "LREAL": 2}
ELEMENTARY = set(INT_RANGE) | set(REAL_DTYPE) | {"BOOL"}

ANY_INT = "ANY_INT"    # type of an untyped integer literal
ANY_REAL = "ANY_REAL"  # type of an untyped real literal


def _family(t: str) -> str:
    if t in INT_RANGE or t == ANY_INT:
        return "int"
    if t in REAL_DTYPE or t == ANY_REAL:
        return "real"
    if t == "BOOL":
        return "bool"
    raise STError(f"unknown type {t!r}")


def _unify(ta: str, tb: str, what: str) -> str:
    """Result type of a binary operation between ta and tb (strict IEC)."""
    fa, fb = _family(ta), _family(tb)
    if fa != fb:
        raise STError(f"{what}: cannot combine {ta} and {tb} without an explicit conversion")
    if ta.startswith("ANY_"):
        return tb
    if tb.startswith("ANY_"):
        return ta
    if fa == "bool":
        return "BOOL"
    rank = INT_RANK if fa == "int" else REAL_RANK
    return ta if rank[ta] >= rank[tb] else tb


def _default_value(t: str):
    if t in INT_RANGE:
        return 0
    if t in REAL_DTYPE:
        return REAL_DTYPE[t](0.0)
    if t == "BOOL":
        return False
    raise STError(f"unsupported type {t!r}")


def _is_lanes(v) -> bool:
    return isinstance(v, np.ndarray)


def lanes(v, n: int) -> np.ndarray:
    """Broadcast a uniform value (or return a per-lane array) as shape (n,)."""
    if isinstance(v, np.ndarray):
        return v
    return np.full(n, v)


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------
KEYWORDS = {
    "IF", "THEN", "ELSIF", "ELSE", "END_IF", "FOR", "TO", "BY", "DO", "END_FOR",
    "CASE", "OF", "END_CASE", "AND", "OR", "XOR", "NOT", "MOD", "TRUE", "FALSE",
    "WHILE", "END_WHILE", "REPEAT", "UNTIL", "END_REPEAT", "EXIT", "RETURN",
    "VAR", "VAR_INPUT", "VAR_OUTPUT", "VAR_IN_OUT", "VAR_TEMP", "VAR_EXTERNAL",
    "VAR_GLOBAL", "END_VAR", "CONSTANT", "RETAIN", "ARRAY", "FUNCTION_BLOCK",
    "END_FUNCTION_BLOCK", "FUNCTION", "END_FUNCTION", "PROGRAM", "END_PROGRAM",
}

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<comment>\(\*.*?\*\))
  | (?P<linecomment>//[^\n]*)
  | (?P<real>\d+(?:_\d+)*\.\d+(?:_\d+)*(?:[eE][+-]?\d+)?)
  | (?P<int>\d+(?:_\d+)*)
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>:=|<=|>=|<>|\.\.|[-+*/()\[\],;:=<>&])
    """,
    re.S | re.X,
)


@dataclass
class Tok:
    kind: str   # 'real' | 'int' | 'ident' | 'kw' | 'op' | 'eof'
    text: str
    line: int
    col: int


def tokenize(text: str) -> List[Tok]:
    toks: List[Tok] = []
    pos, line, line_start = 0, 1, 0
    n = len(text)
    while pos < n:
        m = _TOKEN_RE.match(text, pos)
        if not m:
            snippet = text[pos:pos + 20].replace("\n", "\\n")
            raise STError(f"line {line}: unexpected character(s) {snippet!r}")
        kind = m.lastgroup
        s = m.group(kind)
        if kind not in ("ws", "comment", "linecomment"):
            if kind == "ident":
                up = s.upper()
                toks.append(Tok("kw" if up in KEYWORDS else "ident", up, line, pos - line_start + 1))
            else:
                toks.append(Tok(kind, s, line, pos - line_start + 1))
        nl = s.count("\n")
        if nl:
            line += nl
            line_start = pos + s.rfind("\n") + 1
        pos = m.end()
    if "(*" in text and text.count("(*") != text.count("*)"):
        raise STError("unbalanced (* *) comment")
    toks.append(Tok("eof", "", line, pos - line_start + 1))
    return toks


# --------------------------------------------------------------------------
# AST
# --------------------------------------------------------------------------
@dataclass
class Num:
    value: Union[int, float]
    is_real: bool
    text: str


@dataclass
class BoolLit:
    value: bool


@dataclass
class Ref:
    name: str
    index: Optional[object]  # expression node or None


@dataclass
class Unary:
    op: str
    operand: object


@dataclass
class Bin:
    op: str
    left: object
    right: object


@dataclass
class Call:
    func: str
    args: List[object]


@dataclass
class Assign:
    target: Ref
    expr: object
    line: int


@dataclass
class If:
    branches: List[Tuple[object, list]]
    orelse: Optional[list]


@dataclass
class For:
    var: str
    start: object
    stop: object
    step: Optional[object]
    body: list
    line: int


@dataclass
class Case:
    selector: object
    branches: List[Tuple[List[int], list]]
    orelse: Optional[list]


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
class Parser:
    def __init__(self, text: str):
        self.toks = tokenize(text)
        self.i = 0

    # -- helpers
    @property
    def tok(self) -> Tok:
        return self.toks[self.i]

    def peek(self, k: int = 1) -> Tok:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def at(self, kind: str, text: Optional[str] = None) -> bool:
        t = self.tok
        return t.kind == kind and (text is None or t.text == text)

    def at_kw(self, *words: str) -> bool:
        return self.tok.kind == "kw" and self.tok.text in words

    def expect(self, kind: str, text: Optional[str] = None) -> Tok:
        t = self.tok
        if t.kind != kind or (text is not None and t.text != text):
            want = text or kind
            raise STError(f"line {t.line}:{t.col}: expected {want!r}, found {t.text or t.kind!r}")
        self.i += 1
        return t

    def expect_kw(self, word: str) -> Tok:
        return self.expect("kw", word)

    def expect_op(self, op: str) -> Tok:
        return self.expect("op", op)

    # -- statements
    def parse_statements(self, stop: Sequence[str], in_case: bool = False) -> list:
        stmts = []
        while True:
            t = self.tok
            if t.kind == "eof" or (t.kind == "kw" and t.text in stop):
                return stmts
            if in_case and self._at_case_label():
                return stmts
            if self.at("op", ";"):          # empty statement
                self.i += 1
                continue
            stmts.append(self.parse_statement())
            self.expect_op(";")

    def _at_case_label(self) -> bool:
        t = self.tok
        if t.kind == "op" and t.text == "-" and self.peek().kind == "int":
            nxt = self.peek(2)
        elif t.kind == "int":
            nxt = self.peek()
        else:
            return False
        return nxt.kind == "op" and nxt.text in (":", ",", "..")

    def parse_statement(self):
        t = self.tok
        if t.kind == "kw":
            if t.text == "IF":
                return self.parse_if()
            if t.text == "FOR":
                return self.parse_for()
            if t.text == "CASE":
                return self.parse_case()
            raise STError(f"line {t.line}:{t.col}: unsupported statement {t.text!r}")
        if t.kind == "ident":
            target = self.parse_ref(self.expect("ident"))
            self.expect_op(":=")
            return Assign(target, self.parse_expr(), t.line)
        raise STError(f"line {t.line}:{t.col}: unexpected {t.text!r}")

    def parse_if(self) -> If:
        self.expect_kw("IF")
        cond = self.parse_expr()
        self.expect_kw("THEN")
        branches = [(cond, self.parse_statements(("ELSIF", "ELSE", "END_IF")))]
        orelse = None
        while self.at_kw("ELSIF"):
            self.i += 1
            c = self.parse_expr()
            self.expect_kw("THEN")
            branches.append((c, self.parse_statements(("ELSIF", "ELSE", "END_IF"))))
        if self.at_kw("ELSE"):
            self.i += 1
            orelse = self.parse_statements(("END_IF",))
        self.expect_kw("END_IF")
        return If(branches, orelse)

    def parse_for(self) -> For:
        line = self.expect_kw("FOR").line
        var = self.expect("ident").text
        self.expect_op(":=")
        start = self.parse_expr()
        self.expect_kw("TO")
        stop = self.parse_expr()
        step = None
        if self.at_kw("BY"):
            self.i += 1
            step = self.parse_expr()
        self.expect_kw("DO")
        body = self.parse_statements(("END_FOR",))
        self.expect_kw("END_FOR")
        return For(var, start, stop, step, body, line)

    def _case_int(self) -> int:
        neg = False
        if self.at("op", "-"):
            self.i += 1
            neg = True
        v = int(self.expect("int").text.replace("_", ""))
        return -v if neg else v

    def parse_case(self) -> Case:
        self.expect_kw("CASE")
        sel = self.parse_expr()
        self.expect_kw("OF")
        branches: List[Tuple[List[int], list]] = []
        seen: set = set()
        while self._at_case_label():
            labels: List[int] = []
            while True:
                lo = self._case_int()
                hi = lo
                if self.at("op", ".."):
                    self.i += 1
                    hi = self._case_int()
                for v in range(lo, hi + 1):
                    if v in seen:
                        raise STError(f"duplicate CASE label {v}")
                    seen.add(v)
                    labels.append(v)
                if self.at("op", ","):
                    self.i += 1
                    continue
                break
            self.expect_op(":")
            branches.append((labels, self.parse_statements(("ELSE", "END_CASE"), in_case=True)))
        orelse = None
        if self.at_kw("ELSE"):
            self.i += 1
            orelse = self.parse_statements(("END_CASE",))
        self.expect_kw("END_CASE")
        if not branches:
            raise STError("CASE without labels")
        return Case(sel, branches, orelse)

    def parse_ref(self, ident: Tok) -> Ref:
        if self.at("op", "["):
            self.i += 1
            idx = self.parse_expr()
            self.expect_op("]")
            return Ref(ident.text, idx)
        return Ref(ident.text, None)

    # -- expressions (IEC 61131-3 precedence, lowest first)
    def parse_expr(self):
        return self._parse_or()

    def _parse_or(self):
        left = self._parse_xor()
        while self.at_kw("OR"):
            self.i += 1
            left = Bin("OR", left, self._parse_xor())
        return left

    def _parse_xor(self):
        left = self._parse_and()
        while self.at_kw("XOR"):
            self.i += 1
            left = Bin("XOR", left, self._parse_and())
        return left

    def _parse_and(self):
        left = self._parse_eq()
        while self.at_kw("AND") or self.at("op", "&"):
            self.i += 1
            left = Bin("AND", left, self._parse_eq())
        return left

    def _parse_eq(self):
        left = self._parse_cmp()
        while self.at("op", "=") or self.at("op", "<>"):
            op = self.tok.text
            self.i += 1
            left = Bin(op, left, self._parse_cmp())
        return left

    def _parse_cmp(self):
        left = self._parse_add()
        while self.tok.kind == "op" and self.tok.text in ("<", ">", "<=", ">="):
            op = self.tok.text
            self.i += 1
            left = Bin(op, left, self._parse_add())
        return left

    def _parse_add(self):
        left = self._parse_mul()
        while self.tok.kind == "op" and self.tok.text in ("+", "-"):
            op = self.tok.text
            self.i += 1
            left = Bin(op, left, self._parse_mul())
        return left

    def _parse_mul(self):
        left = self._parse_unary()
        while (self.tok.kind == "op" and self.tok.text in ("*", "/")) or self.at_kw("MOD"):
            op = self.tok.text
            self.i += 1
            left = Bin(op, left, self._parse_unary())
        return left

    def _parse_unary(self):
        if self.at("op", "-"):
            self.i += 1
            return Unary("-", self._parse_unary())
        if self.at("op", "+"):
            self.i += 1
            return self._parse_unary()
        if self.at_kw("NOT"):
            self.i += 1
            return Unary("NOT", self._parse_unary())
        return self._parse_primary()

    def _parse_primary(self):
        t = self.tok
        if t.kind == "real":
            self.i += 1
            return Num(float(t.text.replace("_", "")), True, t.text)
        if t.kind == "int":
            self.i += 1
            return Num(int(t.text.replace("_", "")), False, t.text)
        if t.kind == "kw" and t.text in ("TRUE", "FALSE"):
            self.i += 1
            return BoolLit(t.text == "TRUE")
        if t.kind == "op" and t.text == "(":
            self.i += 1
            e = self.parse_expr()
            self.expect_op(")")
            return e
        if t.kind == "ident":
            self.i += 1
            if self.at("op", "("):
                self.i += 1
                args = [self.parse_expr()]
                while self.at("op", ","):
                    self.i += 1
                    args.append(self.parse_expr())
                self.expect_op(")")
                return Call(t.text, args)
            return self.parse_ref(t)
        raise STError(f"line {t.line}:{t.col}: unexpected {t.text or t.kind!r} in expression")


# --------------------------------------------------------------------------
# Declarations and initialisers
# --------------------------------------------------------------------------
@dataclass
class VarDecl:
    name: str
    type: str
    lo: Optional[int] = None
    hi: Optional[int] = None
    init: Optional[str] = None
    kind: str = "internal"   # 'input' | 'output' | 'internal' | 'constant'
    comment: str = ""

    @property
    def is_array(self) -> bool:
        return self.lo is not None

    @property
    def size(self) -> int:
        return self.hi - self.lo + 1 if self.is_array else 1

    @property
    def writable(self) -> bool:
        return self.kind in ("output", "internal")


def parse_array_size(text: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """IEC 61499 ArraySize: '12' means [0..11]; 'lo..hi' is also accepted (EAE)."""
    if text is None or text.strip() == "":
        return None, None
    s = text.strip()
    m = re.fullmatch(r"\[?\s*(-?\d+)\s*\.\.\s*(-?\d+)\s*\]?", s)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
    elif re.fullmatch(r"\d+", s):
        lo, hi = 0, int(s) - 1
    else:
        raise STError(f"unsupported ArraySize {text!r}")
    if hi < lo:
        raise STError(f"empty array size {text!r}")
    return lo, hi


def _literal_value(tok_list: List[Tok], pos: int, base_type: str):
    """Parse one (optionally signed) literal of base_type at tok_list[pos]."""
    neg = False
    t = tok_list[pos]
    if t.kind == "op" and t.text in ("-", "+"):
        neg = t.text == "-"
        pos += 1
        t = tok_list[pos]
    fam = _family(base_type)
    if fam == "bool":
        if t.kind == "kw" and t.text in ("TRUE", "FALSE") and not neg:
            return t.text == "TRUE", pos + 1
        if t.kind == "int" and t.text in ("0", "1") and not neg:
            return t.text == "1", pos + 1
        raise STError(f"bad BOOL literal {t.text!r}")
    if fam == "int":
        if t.kind != "int":
            raise STError(f"{base_type} initial value needs an integer literal, got {t.text!r}")
        v = int(t.text.replace("_", ""))
        v = -v if neg else v
        lo, hi = INT_RANGE[base_type]
        if not lo <= v <= hi:
            raise STError(f"initial value {v} out of range for {base_type}")
        return v, pos + 1
    if t.kind != "real":
        raise STError(f"{base_type} initial value needs a real literal (e.g. 1.0), got {t.text!r}")
    v = REAL_DTYPE[base_type](float(t.text.replace("_", "")))
    if not np.isfinite(v):
        raise STError(f"literal {t.text} overflows {base_type}")
    return (-v if neg else v), pos + 1


def parse_initializer(text: str, decl: VarDecl):
    """Parse an InitialValue / ':=' initialiser for decl.

    Scalars take one literal. Arrays take '[v, v, n(v), ...]' and the element
    count must equal the array size exactly (strict: a short list is a
    generator bug, EAE only warns WRN_ARRAY_NOINIT)."""
    toks = tokenize(text)
    if not decl.is_array:
        v, pos = _literal_value(toks, 0, decl.type)
        if toks[pos].kind != "eof":
            raise STError(f"{decl.name}: trailing text in initial value {text!r}")
        return v
    if not (toks[0].kind == "op" and toks[0].text == "["):
        raise STError(f"{decl.name}: array initial value must start with '['")
    pos, out = 1, []
    while True:
        t, nxt = toks[pos], toks[pos + 1]
        if t.kind == "int" and nxt.kind == "op" and nxt.text == "(":
            count = int(t.text)
            v, pos = _literal_value(toks, pos + 2, decl.type)
            if not (toks[pos].kind == "op" and toks[pos].text == ")"):
                raise STError(f"{decl.name}: expected ')' in repetition")
            out.extend([v] * count)
            pos += 1
        else:
            v, pos = _literal_value(toks, pos, decl.type)
            out.append(v)
        t = toks[pos]
        if t.kind == "op" and t.text == ",":
            pos += 1
            continue
        if t.kind == "op" and t.text == "]":
            pos += 1
            break
        raise STError(f"{decl.name}: unexpected {t.text!r} in array initial value")
    if toks[pos].kind != "eof":
        raise STError(f"{decl.name}: trailing text after array initial value")
    if len(out) != decl.size:
        raise STError(f"{decl.name}: {len(out)} initial values for an array of {decl.size}")
    if decl.type in REAL_DTYPE:
        return np.array(out, dtype=REAL_DTYPE[decl.type])
    if decl.type == "BOOL":
        return np.array(out, dtype=bool)
    return np.array(out, dtype=np.int64)


# --------------------------------------------------------------------------
# Runtime storage
# --------------------------------------------------------------------------
class ArrayVal:
    """A 1-D IEC array: shape (size,) while uniform, (size, lanes) once a
    lane-dependent value has been written into it."""

    __slots__ = ("data", "lo", "hi", "n")

    def __init__(self, data: np.ndarray, lo: int, n: int):
        self.data = data
        self.lo = lo
        self.hi = lo + data.shape[0] - 1
        self.n = n

    def _promote(self):
        if self.data.ndim == 1:
            self.data = np.repeat(self.data[:, None], self.n, axis=1)

    def get(self, i):
        if _is_lanes(i):
            if i.min() < self.lo or i.max() > self.hi:
                raise STError(f"index out of range [{self.lo}..{self.hi}]")
            k = i - self.lo
            if self.data.ndim == 1:
                return self.data[k]
            return self.data[k, np.arange(self.n)]
        if not self.lo <= i <= self.hi:
            raise STError(f"index {i} out of range [{self.lo}..{self.hi}]")
        return self.data[i - self.lo]

    def set(self, i, v, mask):
        if _is_lanes(i):
            raise STError("lane-dependent index on the left of ':=' is not supported")
        if not self.lo <= i <= self.hi:
            raise STError(f"index {i} out of range [{self.lo}..{self.hi}]")
        k = i - self.lo
        if mask is None and not _is_lanes(v) and self.data.ndim == 1:
            self.data[k] = v
            return
        self._promote()
        if mask is None:
            self.data[k] = v
        else:
            self.data[k] = np.where(mask, v, self.data[k])


# --------------------------------------------------------------------------
# Compiler: AST -> closures
# --------------------------------------------------------------------------
Env = Dict[str, object]
StmtFn = Callable[[Env, Optional[np.ndarray]], None]


class _Lit:
    """Compile-time literal whose concrete type is fixed by its context."""

    __slots__ = ("type", "value")

    def __init__(self, type_: str, value):
        self.type = type_
        self.value = value


class _Expr:
    __slots__ = ("type", "fn")

    def __init__(self, type_: str, fn: Callable[[Env], object]):
        self.type = type_
        self.fn = fn


def _concrete(lit: _Lit, target: str):
    """Convert an untyped literal to a value of concrete type target."""
    fam = _family(target)
    if lit.type == "BOOL" or fam == "bool":
        if lit.type != "BOOL" or fam != "bool":
            raise STError(f"cannot use {lit.type} literal as {target}")
        return bool(lit.value)
    if _family(lit.type) != fam:
        raise STError(f"cannot use {lit.type} literal {lit.value!r} as {target} without conversion")
    if fam == "int":
        lo, hi = INT_RANGE[target]
        if not lo <= lit.value <= hi:
            raise STError(f"literal {lit.value} out of range for {target}")
        return int(lit.value)
    v = REAL_DTYPE[target](lit.value)
    if not np.isfinite(v):
        raise STError(f"literal {lit.value!r} overflows {target}")
    return v


def _as_fn(e, target: str) -> Callable[[Env], object]:
    if isinstance(e, _Lit):
        c = _concrete(e, target)
        return lambda env: c
    return e.fn


def _int_div(a, b):
    if _is_lanes(b):
        if np.any(b == 0):
            raise STError("integer division by zero")
    elif b == 0:
        raise STError("integer division by zero")
    if _is_lanes(a) or _is_lanes(b):
        q = np.abs(a) // np.abs(b)
        return np.where((np.asarray(a) < 0) ^ (np.asarray(b) < 0), -q, q)
    q = abs(a) // abs(b)
    return -q if (a < 0) ^ (b < 0) else q


def _int_mod(a, b):
    return a - b * _int_div(a, b)


_ARITH = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
}
_CMP = {
    "<": lambda a, b: a < b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    ">=": lambda a, b: a >= b,
    "=": lambda a, b: a == b,
    "<>": lambda a, b: a != b,
}


def _bool_and(a, b):
    if _is_lanes(a) or _is_lanes(b):
        return np.logical_and(a, b)
    return bool(a and b)


def _bool_or(a, b):
    if _is_lanes(a) or _is_lanes(b):
        return np.logical_or(a, b)
    return bool(a or b)


def _bool_xor(a, b):
    if _is_lanes(a) or _is_lanes(b):
        return np.logical_xor(a, b)
    return bool(a) != bool(b)


def _check_int(v, t: str, what: str):
    lo, hi = INT_RANGE[t]
    if _is_lanes(v):
        if v.size and (v.min() < lo or v.max() > hi):
            raise STError(f"{what}: value out of range for {t}")
    elif not lo <= v <= hi:
        raise STError(f"{what}: value {v} out of range for {t}")


class Compiler:
    def __init__(self, decls: Dict[str, VarDecl]):
        self.decls = decls

    def decl(self, name: str) -> VarDecl:
        d = self.decls.get(name)
        if d is None:
            raise STError(f"variable {name!r} is not declared")
        return d

    # ---- expressions
    def expr(self, node):
        if isinstance(node, Num):
            return _Lit(ANY_REAL if node.is_real else ANY_INT, node.value)
        if isinstance(node, BoolLit):
            return _Lit("BOOL", node.value)
        if isinstance(node, Ref):
            return self._ref(node)
        if isinstance(node, Unary):
            return self._unary(node)
        if isinstance(node, Bin):
            return self._bin(node)
        if isinstance(node, Call):
            return self._call(node)
        raise STError(f"unsupported expression {node!r}")

    def _index_fn(self, node) -> Callable[[Env], object]:
        e = self.expr(node)
        if _family(e.type) != "int":
            raise STError(f"array index must be an integer, got {e.type}")
        if isinstance(e, _Lit):
            v = int(e.value)
            return lambda env: v
        return e.fn

    def _ref(self, node: Ref) -> _Expr:
        d = self.decl(node.name)
        name = node.name
        if node.index is None:
            if d.is_array:
                raise STError(f"array {name} used without an index")
            return _Expr(d.type, lambda env: env[name])
        if not d.is_array:
            raise STError(f"{name} is not an array")
        idx = self._index_fn(node.index)
        return _Expr(d.type, lambda env: env[name].get(idx(env)))

    def _unary(self, node: Unary):
        e = self.expr(node.operand)
        if node.op == "-":
            if _family(e.type) == "bool":
                raise STError("unary minus on BOOL")
            if isinstance(e, _Lit):
                return _Lit(e.type, -e.value)
            f = e.fn
            return _Expr(e.type, lambda env: -f(env))
        if node.op == "NOT":
            if _family(e.type) != "bool":
                raise STError(f"NOT on {e.type} (bit strings are not supported)")
            if isinstance(e, _Lit):
                return _Lit("BOOL", not e.value)
            f = e.fn

            def _not(env):
                v = f(env)
                return np.logical_not(v) if _is_lanes(v) else (not v)

            return _Expr("BOOL", _not)
        raise STError(f"unsupported unary operator {node.op}")

    def _bin(self, node: Bin):
        a, b = self.expr(node.left), self.expr(node.right)
        op = node.op
        t = _unify(a.type, b.type, f"operator {op}")
        fam = _family(t)
        if isinstance(a, _Lit) and isinstance(b, _Lit):
            raise STError(f"constant expression '{op}' between two literals is not supported")
        if t.startswith("ANY_"):  # unreachable: one side is concrete
            raise STError("internal typing error")
        fa, fb = _as_fn(a, t), _as_fn(b, t)
        if op in ("AND", "OR", "XOR"):
            if fam != "bool":
                raise STError(f"{op} on {t}: bit strings are not supported")
            g = {"AND": _bool_and, "OR": _bool_or, "XOR": _bool_xor}[op]
            return _Expr("BOOL", lambda env: g(fa(env), fb(env)))
        if op in _CMP:
            if fam == "bool" and op not in ("=", "<>"):
                raise STError(f"ordering comparison {op} on BOOL")
            g = _CMP[op]
            return _Expr("BOOL", lambda env: g(fa(env), fb(env)))
        if fam == "bool":
            raise STError(f"arithmetic {op} on BOOL")
        if op in _ARITH:
            g = _ARITH[op]
            if fam == "real":
                dt = REAL_DTYPE[t]

                def _real_op(env, g=g, dt=dt):
                    r = g(fa(env), fb(env))
                    return r if r.dtype == dt else r.astype(dt)

                return _Expr(t, _real_op)
            return _Expr(t, lambda env: g(fa(env), fb(env)))
        if op == "/":
            if fam == "int":
                return _Expr(t, lambda env: _int_div(fa(env), fb(env)))

            def _rdiv(env):
                d = fb(env)
                if (np.any(d == 0) if _is_lanes(d) else d == 0):
                    raise STError("REAL division by zero")
                return fa(env) / d

            return _Expr(t, _rdiv)
        if op == "MOD":
            if fam != "int":
                raise STError("MOD needs integer operands")
            return _Expr(t, lambda env: _int_mod(fa(env), fb(env)))
        raise STError(f"unsupported operator {op}")

    def _call(self, node: Call):
        f = node.func
        args = [self.expr(a) for a in node.args]
        if f in ("MAX", "MIN"):
            if len(args) < 2:
                raise STError(f"{f} needs at least two arguments")
            t = args[0].type
            for a in args[1:]:
                t = _unify(t, a.type, f)
            if t.startswith("ANY_") or _family(t) == "bool":
                raise STError(f"{f} needs a typed numeric argument")
            fns = [_as_fn(a, t) for a in args]
            npf = np.maximum if f == "MAX" else np.minimum
            pyf = max if f == "MAX" else min

            def _mm(env):
                vals = [g(env) for g in fns]
                if any(_is_lanes(v) for v in vals) or _family(t) == "real":
                    r = vals[0]
                    for v in vals[1:]:
                        r = npf(r, v)
                    return r
                return pyf(vals)

            return _Expr(t, _mm)
        if f == "ABS":
            if len(args) != 1 or isinstance(args[0], _Lit) or _family(args[0].type) == "bool":
                raise STError("ABS needs one typed numeric argument")
            g = args[0].fn
            return _Expr(args[0].type, lambda env: abs(g(env)))
        raise STError(f"unsupported function {f}()")

    # ---- statements
    def stmts(self, nodes: list) -> StmtFn:
        fns = [self.stmt(n) for n in nodes]

        def run(env, mask):
            for g in fns:
                g(env, mask)

        return run

    def stmt(self, node) -> StmtFn:
        if isinstance(node, Assign):
            return self._assign(node)
        if isinstance(node, If):
            return self._if(node)
        if isinstance(node, For):
            return self._for(node)
        if isinstance(node, Case):
            return self._case(node)
        raise STError(f"unsupported statement {node!r}")

    def _store_fn(self, d: VarDecl, e, line: int) -> Callable[[Env], object]:
        """Value function for an assignment to d, with IEC type rules."""
        what = f"line {line}: {d.name} :="
        if isinstance(e, _Lit):
            c = _concrete(e, d.type)
            return lambda env: c
        fs, ft = _family(e.type), _family(d.type)
        if fs != ft:
            raise STError(f"{what} cannot assign {e.type} to {d.type} without conversion")
        if fs == "int" and INT_RANK[e.type] > INT_RANK[d.type]:
            raise STError(f"{what} narrowing {e.type} to {d.type} needs an explicit conversion")
        if fs == "real" and REAL_RANK[e.type] > REAL_RANK[d.type]:
            raise STError(f"{what} narrowing {e.type} to {d.type} needs an explicit conversion")
        f = e.fn
        if fs == "real" and e.type != d.type:
            dt = REAL_DTYPE[d.type]
            return lambda env: np.asarray(f(env)).astype(dt)[()]
        if fs == "int":
            t = d.type

            def _checked(env):
                v = f(env)
                _check_int(v, t, what)
                return v

            return _checked
        return f

    def _assign(self, node: Assign) -> StmtFn:
        d = self.decl(node.target.name)
        if not d.writable:
            raise STError(f"line {node.line}: {d.name} is read-only ({d.kind})")
        value = self._store_fn(d, self.expr(node.expr), node.line)
        name = d.name
        if node.target.index is None:
            if d.is_array:
                raise STError(f"line {node.line}: whole-array assignment to {name} is not supported")

            def run(env, mask):
                v = value(env)
                if isinstance(v, np.ndarray) and v.base is not None:
                    v = v.copy()
                env[name] = v if mask is None else np.where(mask, v, env[name])

            return run
        if not d.is_array:
            raise STError(f"line {node.line}: {name} is not an array")
        idx = self._index_fn(node.target.index)

        def run_idx(env, mask):
            env[name].set(idx(env), value(env), mask)

        return run_idx

    def _cond_fn(self, node) -> Callable[[Env], object]:
        e = self.expr(node)
        if e.type != "BOOL":
            raise STError(f"condition must be BOOL, got {e.type}")
        if isinstance(e, _Lit):
            v = bool(e.value)
            return lambda env: v
        return e.fn

    def _if(self, node: If) -> StmtFn:
        branches = [(self._cond_fn(c), self.stmts(b)) for c, b in node.branches]
        orelse = self.stmts(node.orelse) if node.orelse is not None else None

        def run(env, mask):
            remaining = mask
            for cond, body in branches:
                c = cond(env)
                if not _is_lanes(c):
                    if c:
                        body(env, remaining)
                        return
                    continue
                take = c if remaining is None else (remaining & c)
                if take.any():
                    body(env, None if take.all() else take)
                remaining = ~c if remaining is None else (remaining & ~c)
                if not remaining.any():
                    return
            if orelse is not None:
                orelse(env, remaining)

        return run

    def _for(self, node: For) -> StmtFn:
        d = self.decl(node.var)
        if d.is_array or d.type not in INT_RANGE:
            raise STError(f"line {node.line}: FOR control variable {d.name} must be an integer scalar")
        if not d.writable:
            raise STError(f"line {node.line}: FOR control variable {d.name} is read-only")
        start = self._store_fn(d, self.expr(node.start), node.line)
        stop = self._store_fn(d, self.expr(node.stop), node.line)
        step = self._store_fn(d, self.expr(node.step), node.line) if node.step is not None else (lambda env: 1)
        body = self.stmts(node.body)
        name, lo, hi = d.name, *INT_RANGE[d.type]

        def run(env, mask):
            if mask is not None:
                raise STError(f"line {node.line}: FOR inside a lane-dependent branch is not supported")
            s, e, st = start(env), stop(env), step(env)
            if _is_lanes(s) or _is_lanes(e) or _is_lanes(st):
                raise STError(f"line {node.line}: lane-dependent FOR bounds are not supported")
            if st == 0:
                raise STError(f"line {node.line}: FOR step is 0")
            v = s
            while (v <= e) if st > 0 else (v >= e):
                env[name] = v
                body(env, None)
                cur = env[name]
                if _is_lanes(cur) or cur != v:
                    raise STError(f"line {node.line}: FOR control variable {name} modified in the loop body")
                v += st
                if not lo <= v <= hi:
                    # on a PLC the control variable would wrap and the loop never ends
                    raise STError(f"line {node.line}: FOR control variable {name} overflows {d.type}")
            env[name] = v

        return run

    def _case(self, node: Case) -> StmtFn:
        sel_e = self.expr(node.selector)
        if _family(sel_e.type) != "int" or isinstance(sel_e, _Lit):
            raise STError("CASE selector must be a typed integer expression")
        _check_int(np.array([v for labels, _ in node.branches for v in labels]), sel_e.type, "CASE label")
        sel = sel_e.fn
        branches = [(labels, np.array(labels), self.stmts(b)) for labels, b in node.branches]
        orelse = self.stmts(node.orelse) if node.orelse is not None else None

        def run(env, mask):
            s = sel(env)
            if not _is_lanes(s):
                for labels, _, body in branches:
                    if s in labels:
                        body(env, mask)
                        return
                if orelse is not None:
                    orelse(env, mask)
                return
            remaining = mask
            for _, arr, body in branches:
                m = np.isin(s, arr)
                take = m if remaining is None else (remaining & m)
                if take.any():
                    body(env, None if take.all() else take)
                remaining = ~m if remaining is None else (remaining & ~m)
            if orelse is not None and remaining.any():
                orelse(env, remaining)

        return run


def compile_body(text: str, decls: Dict[str, VarDecl]) -> StmtFn:
    """Parse and compile a statement list (an algorithm or a POU body)."""
    p = Parser(text)
    stmts = p.parse_statements(())
    if p.tok.kind != "eof":
        t = p.tok
        raise STError(f"line {t.line}:{t.col}: unexpected {t.text!r}")
    return Compiler(decls).stmts(stmts)


def _initial_env(decls: Dict[str, VarDecl], n: int) -> Env:
    env: Env = {}
    for d in decls.values():
        if d.type not in ELEMENTARY:
            raise STError(f"{d.name}: unsupported type {d.type}")
        if d.is_array:
            if d.init is not None:
                data = parse_initializer(d.init, d)
            elif d.type in REAL_DTYPE:
                data = np.zeros(d.size, dtype=REAL_DTYPE[d.type])
            elif d.type == "BOOL":
                data = np.zeros(d.size, dtype=bool)
            else:
                data = np.zeros(d.size, dtype=np.int64)
            env[d.name] = ArrayVal(data, d.lo, n)
        else:
            env[d.name] = parse_initializer(d.init, d) if d.init is not None else _default_value(d.type)
    return env


def _set_input(env: Env, d: VarDecl, value, n: int):
    """Write an externally supplied input value (uniform or per-lane)."""
    if d.is_array:
        arr = np.asarray(value)
        if arr.shape not in ((d.size,), (d.size, n)):
            raise STError(f"{d.name}: expected shape ({d.size},) or ({d.size}, {n}), got {arr.shape}")
        if d.type in REAL_DTYPE:
            arr = arr.astype(REAL_DTYPE[d.type])
        env[d.name] = ArrayVal(arr.copy(), d.lo, n)
        return
    if d.type in REAL_DTYPE:
        dt = REAL_DTYPE[d.type]
        v = np.asarray(value).astype(dt)
        if v.shape not in ((), (n,)):
            raise STError(f"{d.name}: expected a scalar or shape ({n},), got {v.shape}")
        env[d.name] = v[()] if v.shape == () else v.copy()
        return
    if d.type in INT_RANGE:
        v = np.asarray(value)
        if not np.issubdtype(v.dtype, np.integer):
            raise STError(f"{d.name}: integer input expected")
        _check_int(v.astype(np.int64), d.type, d.name)
        env[d.name] = int(v) if v.shape == () else v.astype(np.int64)
        return
    v = np.asarray(value).astype(bool)
    env[d.name] = bool(v) if v.shape == () else v


def _snapshot(v):
    if isinstance(v, ArrayVal):
        return v.data.copy()
    if isinstance(v, np.ndarray):
        return v.copy()
    return v


# --------------------------------------------------------------------------
# IEC 61499 Basic FB
# --------------------------------------------------------------------------
@dataclass
class Transition:
    source: str
    destination: str
    condition: str


class BasicFB:
    """Executes an IEC 61499 Basic FB type for ``n`` lanes.

    ``fbt`` is a ``Path`` to a .fbt file, or the XML text itself (``str``)."""

    def __init__(self, fbt: Union[str, Path], n: int = 1):
        text = fbt.read_text(encoding="utf-8") if isinstance(fbt, Path) else fbt
        root = ET.fromstring(text.encode("utf-8"))
        if root.tag != "FBType":
            raise STError(f"root element is {root.tag}, expected FBType")
        self.name = root.get("Name")
        self.n = n
        il = root.find("InterfaceList")
        if il is None:
            raise STError("missing InterfaceList")
        basic = root.find("BasicFB")
        if basic is None:
            raise STError("not a Basic FB (no BasicFB element)")

        def events(tag: str) -> Dict[str, List[str]]:
            out: Dict[str, List[str]] = {}
            for e in il.findall(f"{tag}/Event"):
                nm = e.get("Name").upper()
                if nm in out:
                    raise STError(f"duplicate event {nm}")
                out[nm] = [w.get("Var").upper() for w in e.findall("With")]
            return out

        self.event_inputs = events("EventInputs")
        self.event_outputs = events("EventOutputs")

        self.decls: Dict[str, VarDecl] = {}
        for tag, kind in (("InputVars", "input"), ("OutputVars", "output")):
            for v in il.findall(f"{tag}/VarDeclaration"):
                self._add_decl(v, kind)
        for v in basic.findall("InternalVars/VarDeclaration"):
            self._add_decl(v, "internal")

        for ev, withs in self.event_inputs.items():
            for w in withs:
                if w not in self.decls or self.decls[w].kind != "input":
                    raise STError(f"event input {ev} WITH {w}: not a data input")
        for ev, withs in self.event_outputs.items():
            for w in withs:
                if w not in self.decls or self.decls[w].kind != "output":
                    raise STError(f"event output {ev} WITH {w}: not a data output")

        ecc = basic.find("ECC")
        if ecc is None:
            raise STError("Basic FB without ECC")
        self.states: List[str] = []
        self.actions: Dict[str, List[Tuple[Optional[str], Optional[str]]]] = {}
        for s in ecc.findall("ECState"):
            nm = s.get("Name")
            if nm in self.actions:
                raise STError(f"duplicate EC state {nm}")
            self.states.append(nm)
            self.actions[nm] = [(a.get("Algorithm"), a.get("Output")) for a in s.findall("ECAction")]
        self.transitions = [Transition(t.get("Source"), t.get("Destination"), t.get("Condition").strip())
                            for t in ecc.findall("ECTransition")]
        if not self.states:
            raise STError("ECC has no states")

        self.algorithm_text: Dict[str, str] = {}
        for a in basic.findall("Algorithm"):
            st = a.find("ST")
            if st is None:
                raise STError(f"algorithm {a.get('Name')} is not ST")
            body = st.text if (st.text or "").strip() else (st.get("Text") or "")
            self.algorithm_text[a.get("Name")] = body
        self.algorithms = {nm: compile_body(txt, self.decls) for nm, txt in self.algorithm_text.items()}

        for nm, acts in self.actions.items():
            for alg, out in acts:
                if alg and alg not in self.algorithms:
                    raise STError(f"state {nm}: unknown algorithm {alg}")
                if out and out.upper() not in self.event_outputs:
                    raise STError(f"state {nm}: unknown output event {out}")
        for t in self.transitions:
            if t.source not in self.actions or t.destination not in self.actions:
                raise STError(f"transition {t.source}->{t.destination}: unknown state")
            self._condition_event(t.condition)  # validates the syntax

        self.reset()

    def _add_decl(self, v: ET.Element, kind: str):
        nm = v.get("Name").upper()
        if nm in self.decls:
            raise STError(f"duplicate variable {nm}")
        lo, hi = parse_array_size(v.get("ArraySize"))
        self.decls[nm] = VarDecl(nm, v.get("Type").upper(), lo, hi, v.get("InitialValue"), kind,
                                 v.get("Comment") or "")

    def _condition_event(self, cond: str) -> Optional[str]:
        """Return the event a transition condition waits for (None = guard '1')."""
        c = cond.strip().upper()
        if c in ("1", "TRUE"):
            return None
        if c in self.event_inputs:
            return c
        raise STError(f"unsupported ECC transition condition {cond!r} (only EVENT or 1)")

    def reset(self):
        """Cold start: initial values, initial EC state (the first one)."""
        self.env = _initial_env(self.decls, self.n)
        self.state = self.states[0]

    def get(self, name: str):
        """Current value of a variable (arrays as numpy: (size,) or (size, n))."""
        return _snapshot(self.env[name.upper()])

    def fire(self, event: str, inputs: Optional[Dict[str, object]] = None, max_steps: int = 100):
        """Deliver an input event. Returns the emitted output events as a list
        of (event name, {output var: value sampled at emission})."""
        ev = event.upper()
        if ev not in self.event_inputs:
            raise STError(f"{event} is not an event input of {self.name}")
        inputs = {k.upper(): v for k, v in (inputs or {}).items()}
        unknown = set(inputs) - {n for n, d in self.decls.items() if d.kind == "input"}
        if unknown:
            raise STError(f"not data inputs: {sorted(unknown)}")
        for w in self.event_inputs[ev]:           # WITH-latching
            if w not in inputs:
                raise STError(f"event {ev} latches {w} but no value was supplied")
            _set_input(self.env, self.decls[w], inputs[w], self.n)
        emitted: List[Tuple[str, Dict[str, object]]] = []
        pending: Optional[str] = ev
        for _ in range(max_steps):
            fired = None
            for t in self.transitions:
                if t.source != self.state:
                    continue
                need = self._condition_event(t.condition)
                if need is None or need == pending:
                    fired = t
                    break
            pending = None                         # the event is consumed
            if fired is None:
                return emitted
            self.state = fired.destination
            for alg, out in self.actions[self.state]:
                if alg:
                    self.algorithms[alg](self.env, None)
                if out:
                    o = out.upper()
                    emitted.append((o, {w: _snapshot(self.env[w]) for w in self.event_outputs[o]}))
        raise STError(f"ECC did not settle within {max_steps} transitions")


# --------------------------------------------------------------------------
# IEC 61131-3 FUNCTION_BLOCK (.st)
# --------------------------------------------------------------------------
_SECTION_KIND = {"VAR_INPUT": "input", "VAR_OUTPUT": "output", "VAR": "internal"}


def _parse_declarations(p: Parser) -> Dict[str, VarDecl]:
    decls: Dict[str, VarDecl] = {}
    while p.at_kw("VAR_INPUT", "VAR_OUTPUT", "VAR"):
        section = p.tok.text
        p.i += 1
        kind = _SECTION_KIND[section]
        if p.at_kw("CONSTANT"):
            if section != "VAR":
                raise STError(f"{section} CONSTANT is not supported")
            p.i += 1
            kind = "constant"
        while not p.at_kw("END_VAR"):
            names = [p.expect("ident").text]
            while p.at("op", ","):
                p.i += 1
                names.append(p.expect("ident").text)
            p.expect_op(":")
            lo = hi = None
            if p.at_kw("ARRAY"):
                p.i += 1
                p.expect_op("[")
                lo = p._case_int()
                p.expect_op("..")
                hi = p._case_int()
                p.expect_op("]")
                p.expect_kw("OF")
            tname = p.expect("ident").text
            init = None
            if p.at("op", ":="):
                p.i += 1
                start = p.i
                depth = 0
                while not (depth == 0 and p.at("op", ";")):
                    if p.tok.kind == "eof":
                        raise STError("unterminated initialiser")
                    if p.at("op", "[") or p.at("op", "("):
                        depth += 1
                    elif p.at("op", "]") or p.at("op", ")"):
                        depth -= 1
                    p.i += 1
                init = _detokenize(p.toks[start:p.i])
            p.expect_op(";")
            if kind == "constant" and init is None:
                raise STError(f"constant {names} without initial value")
            for nm in names:
                if nm in decls:
                    raise STError(f"duplicate variable {nm}")
                decls[nm] = VarDecl(nm, tname, lo, hi, init, kind)
        p.expect_kw("END_VAR")
    return decls


def _detokenize(toks: List[Tok]) -> str:
    return " ".join(t.text for t in toks)


class STFunctionBlock:
    """Executes an IEC 61131-3 FUNCTION_BLOCK POU for ``n`` lanes.

    ``st`` is a ``Path`` to a .st file, or the source text itself (``str``).
    Calling the instance sets the given VAR_INPUTs, runs the body once and
    returns the VAR_OUTPUTs."""

    def __init__(self, st: Union[str, Path], n: int = 1):
        text = st.read_text(encoding="utf-8") if isinstance(st, Path) else st
        p = Parser(text)
        p.expect_kw("FUNCTION_BLOCK")
        self.name = p.expect("ident").text
        self.decls = _parse_declarations(p)
        body = p.parse_statements(("END_FUNCTION_BLOCK",))
        p.expect_kw("END_FUNCTION_BLOCK")
        if p.tok.kind != "eof":
            raise STError(f"text after END_FUNCTION_BLOCK: {p.tok.text!r}")
        self.n = n
        self.body = Compiler(self.decls).stmts(body)
        self.reset()

    def reset(self):
        self.env = _initial_env(self.decls, self.n)

    def get(self, name: str):
        return _snapshot(self.env[name.upper()])

    def __call__(self, **inputs) -> Dict[str, object]:
        for k, v in inputs.items():
            d = self.decls.get(k.upper())
            if d is None or d.kind != "input":
                raise STError(f"{k} is not a VAR_INPUT of {self.name}")
            _set_input(self.env, d, v, self.n)
        self.body(self.env, None)
        return {nm: _snapshot(self.env[nm]) for nm, d in self.decls.items() if d.kind == "output"}

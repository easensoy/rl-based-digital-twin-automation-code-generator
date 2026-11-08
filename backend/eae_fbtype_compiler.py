"""
Compile a .fbt with EcoStruxure Automation Expert's own FB type compiler.

``FBTypeCompiler.exe`` ("IEC 61499 FB type compiler", EAE Buildtime
``Studio\\bin\\x64``) parses the FB type, type-checks its ST algorithms and
writes a ``<Name>.bin``. Run standalone, outside an EAE solution, it reports two
messages that come from the missing solution context. Both also appear for
Schneider Electric's own library type ``SE.App2WWW/rotationArrayValidation.fbt``,
and neither stops it from writing the .bin or returning exit code 0:

* ``ERR_XML_SNAPSHOT_GUID_INCONSISTENT`` (no base snapshot for a
  non-incremental compile), once per type;
* ``ERR_SYNTAX: '[...]' is not correct structured text syntax`` for every
  array ``InitialValue``. The XML stage itself does read the initialiser: a
  short list draws ``WRN_ARRAY_NOINIT``.

Real problems were checked to give exit code 1 and no .bin: an undefined
variable (ERR_NO_SUCH_VAR), REAL assigned to INT (ERR_CAST_UNDEFINED), a
syntax error, and a constant index out of range (ERR_RANGE). An InitialValue
with 6,500 or more elements crashed the compiler with a stack overflow
(exit code 0xC00000FD, empty .bin). Those cases are reported as failures here.

CompileReport keeps every message line as the compiler printed it, in three
groups: errors, warnings, and the known standalone-compile messages above.
``ok`` means only that the FB type compiler accepted the type; it says nothing
about importing it into an EAE solution or running it on a controller.
"""

from __future__ import annotations

import os
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

_BUILDTIME_GLOB = "EcoStruxure Automation Expert - Buildtime *"
_RELATIVE_EXE = Path("Studio") / "bin" / "x64" / "FBTypeCompiler.exe"
_MSG = re.compile(r"\b(error|warning)\s+([A-Z][A-Z0-9_]*):\s?(.*)$")
_INIT_SYNTAX = re.compile(r"^'(.*)' is not correct structured text syntax\s*$", re.S)


def find_fbtype_compiler() -> Optional[Path]:
    """$EAE_FBTYPECOMPILER, else the newest EAE Buildtime under Program Files."""
    env = os.environ.get("EAE_FBTYPECOMPILER")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    roots = {Path(os.environ.get(v, d)) for v, d in (("ProgramFiles", r"C:\Program Files"),
                                                     ("ProgramW6432", r"C:\Program Files"))}
    found = []
    for root in roots:
        base = root / "Schneider Electric"
        if not base.is_dir():
            continue
        for d in base.glob(_BUILDTIME_GLOB):
            exe = d / _RELATIVE_EXE
            if exe.is_file():
                ver = tuple(int(x) for x in re.findall(r"\d+", d.name.rsplit(" ", 1)[-1]))
                found.append((ver, exe))
    return max(found)[1] if found else None


@dataclass
class CompileReport:
    exe: Path
    args: List[str]
    returncode: int
    bin_path: Optional[Path]
    bin_bytes: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    context_messages: List[str] = field(default_factory=list)
    output: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.bin_bytes > 0 and not self.errors and not self.warnings

    def summary(self) -> str:
        version = self.exe.parents[3].name.rsplit(" ", 1)[-1] if len(self.exe.parents) > 3 else "?"
        state = "accepted the type" if self.ok else "REJECTED the type"
        name = self.bin_path.name if self.bin_path else "no .bin"
        lines = [f"EAE {version} FBTypeCompiler {' '.join(self.args) or '(default options)'} {state}: "
                 f"exit code {self.returncode:#x}, {name} {self.bin_bytes:,} B, {len(self.errors)} errors, "
                 f"{len(self.warnings)} warnings, {len(self.context_messages)} known standalone-compile messages"]
        lines += ["  " + m[:300] for m in self.errors + self.warnings]  # shortened here; the report keeps them whole
        return "\n".join(lines)


def _array_initial_values(fbt: Path) -> set:
    root = ET.parse(fbt).getroot()
    return {v.get("InitialValue") for v in root.iter("VarDeclaration")
            if v.get("ArraySize") and v.get("InitialValue")}


def classify(output: str, array_initial_values: set):
    """Split compiler output into (errors, warnings, known standalone-compile
    messages), each a list of the message lines as printed (whitespace-stripped)."""
    errors, warnings, context = [], [], []
    for line in output.splitlines():
        m = _MSG.search(line)
        if not m:
            continue
        level, code, text = m.groups()
        if code == "ERR_XML_SNAPSHOT_GUID_INCONSISTENT":
            context.append(line.strip())
            continue
        if code == "ERR_SYNTAX":
            q = _INIT_SYNTAX.match(text)
            if q and q.group(1) in array_initial_values:
                context.append(line.strip())
                continue
        (errors if level == "error" else warnings).append(line.strip())
    return errors, warnings, context


def compile_fbt(fbt, out_dir, exe: Optional[Path] = None, extra_args: Sequence[str] = (),
                timeout: float = 600) -> CompileReport:
    """Compile one .fbt into out_dir and classify the compiler's messages."""
    fbt = Path(fbt).resolve()
    exe = exe or find_fbtype_compiler()
    if exe is None:
        raise FileNotFoundError("FBTypeCompiler.exe not found; set EAE_FBTYPECOMPILER")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / (fbt.stem + ".bin")
    if bin_path.exists():
        bin_path.unlink()
    cmd = [str(exe), *extra_args, f"/outputDir:{out_dir}", str(fbt)]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout)
    output = (r.stdout or "") + (r.stderr or "")
    errors, warnings, context = classify(output, _array_initial_values(fbt))
    size = bin_path.stat().st_size if bin_path.exists() else 0
    return CompileReport(Path(exe), list(extra_args), r.returncode, bin_path if bin_path.exists() else None,
                         size, errors, warnings, context, output)

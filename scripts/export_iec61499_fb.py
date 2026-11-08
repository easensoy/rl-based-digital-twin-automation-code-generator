"""
Export a trained DQN policy as PLC code that executes the network:
an IEC 61499 Basic FB for EcoStruxure Automation Expert (<NAME>.fbt) and an
IEC 61131-3 FUNCTION_BLOCK (<NAME>.st).

Usage (from the repo root, Windows):
    .venv\\Scripts\\python.exe scripts\\export_iec61499_fb.py ^
        visual_outputs\\run\\policy_snapshot_20261106_190932.pt --verify 10000 --eae-check

Writes to data\\generated_code\\iec61499\\ by default. --verify N replays N
AutomationEnvironment rollout states through the generated .fbt (interpreted)
and compares with the PyTorch network; --eae-check compiles the .fbt with
EAE's FBTypeCompiler.exe when EAE Buildtime is installed. Options: --help.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.iec61499_exporter import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

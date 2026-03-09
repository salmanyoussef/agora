"""
Reusable REPL/RLM setup: read instructions from pyproject.toml [tool.agora.repl]
and install Deno if missing so the Technical Agent can use DSPy's RLM.

Usage:
  From backend root:  uv run python -m app.scripts.setup_repl
  After install:      agora-setup-repl

Install optional deps (for TOML parsing on Python 3.10):  pip install -e ".[rlm]"
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _find_pyproject() -> Path | None:
    """Locate pyproject.toml: cwd, then backend root (parent of app)."""
    cwd = Path.cwd()
    if (cwd / "pyproject.toml").exists():
        return cwd / "pyproject.toml"
    # When run as script from backend root, __file__ is app/scripts/setup_repl.py
    app_scripts = Path(__file__).resolve().parent
    backend_root = app_scripts.parent.parent
    if (backend_root / "pyproject.toml").exists():
        return backend_root / "pyproject.toml"
    return None


def _load_repl_config() -> dict:
    """Load [tool.agora.repl] from pyproject.toml. Fallback to defaults if missing."""
    defaults = {
        "description": "Technical Agent RLM uses a sandboxed Python REPL (Deno + Pyodide).",
        "doc_url": "https://docs.deno.com/runtime/getting_started/installation",
        "install_unix": "curl -fsSL https://deno.land/install.sh | sh",
        "install_windows": "irm https://deno.land/install.ps1 | iex",
        "system_prerequisites_unix": [
            "unzip or 7z (Deno install script needs one to extract the binary)",
            "Ubuntu/Debian:  sudo apt install unzip",
            "Fedora/RHEL:    sudo dnf install unzip",
            "Alpine:         sudo apk add unzip",
            "macOS:          brew install unzip",
        ],
    }
    path = _find_pyproject()
    if not path:
        return defaults
    try:
        with open(path, "rb") as f:
            data = _parse_toml(f.read())
        return {**defaults, **(data.get("tool", {}).get("agora", {}).get("repl") or {})}
    except Exception:
        return defaults


def _parse_toml(data: bytes) -> dict:
    """Parse TOML using tomllib (3.11+) or tomli."""
    if sys.version_info >= (3, 11):
        import tomllib
        return tomllib.loads(data.decode("utf-8"))
    try:
        import tomli
        return tomli.loads(data.decode("utf-8"))
    except ImportError:
        # No tomli on 3.10: return empty so we use defaults
        return {}


def _has_unzip_or_7z() -> bool:
    """Deno install script needs unzip or 7z to extract the binary."""
    return bool(shutil.which("unzip") or shutil.which("7z") or shutil.which("7za"))


def main() -> int:
    """Check for Deno; if missing, run install from [tool.agora.repl]."""
    config = _load_repl_config()
    if shutil.which("deno"):
        print("Deno is already installed. Technical Agent RLM (REPL) is available.")
        return 0
    print("Deno not found. The Technical Agent will fall back to Predict unless Deno is installed.")
    print()
    print(config.get("description", ""))
    print(f"Docs: {config.get('doc_url', '')}")
    print()
    if os.name == "nt":
        print("On Windows (PowerShell), run:")
        print(f"  {config.get('install_windows', '')}")
        print()
        print("Or install via winget:  winget install Deno.Land.Deno")
        return 0
    if not _has_unzip_or_7z():
        prereqs = config.get("system_prerequisites_unix")
        if isinstance(prereqs, list):
            print("System prerequisites (install via OS package manager), then run this again:", file=sys.stderr)
            for line in prereqs:
                print(f"  {line}", file=sys.stderr)
        else:
            print(
                "The Deno install script requires unzip or 7z. Install one (e.g. sudo apt install unzip), then run this again.",
                file=sys.stderr,
            )
        print("  See: https://github.com/denoland/deno_install#either-unzip-or-7z-is-required", file=sys.stderr)
        return 1
    cmd = config.get("install_unix", "curl -fsSL https://deno.land/install.sh | sh")
    print("Running install (Unix):")
    print(f"  {cmd}")
    print()
    try:
        subprocess.run(cmd, shell=True, check=True)
        print()
        print("Install finished. Restart your shell or run:  export PATH=\"$HOME/.deno/bin:$PATH\"")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Install failed: {e}", file=sys.stderr)
        if e.returncode != 0:
            print(
                "If the error was about unzip/7z, install one (e.g. sudo apt install unzip) and try again.",
                file=sys.stderr,
            )
        return e.returncode if e.returncode else 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

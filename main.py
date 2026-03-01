"""
InsightX — main.py
Auto-installer bootstrap: checks for required packages, prompts the user,
and installs anything missing before launching the app.
"""

import sys
import subprocess
import importlib
import os

# ─────────────────────────────────────────────────────────────────────────────
# Package map: { import_name: pip_install_name }
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_PACKAGES = {
    "tkinter":        None,          # built-in, special handling
    "pandas":         "pandas",
    "numpy":          "numpy",
    "matplotlib":     "matplotlib",
    "transformers":   "transformers",
    "torch":          "torch",
    "bitsandbytes":   "bitsandbytes",
    "accelerate":     "accelerate",
    "sentencepiece":  "sentencepiece",
    "protobuf":       "protobuf",
}

# Optional — warn if missing but don't block launch
OPTIONAL_PACKAGES = {
    "flash_attn": "flash-attn",
}


def _check_import(import_name):
    """Return True if the module can be imported."""
    if import_name == "tkinter":
        try:
            import tkinter  # noqa: F401
            return True
        except ImportError:
            return False
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def _pip_install(pip_name):
    """Run pip install for the given package. Returns (success, output_str)."""
    cmd = [sys.executable, "-m", "pip", "install", pip_name, "--quiet"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Installation timed out after 5 minutes."
    except Exception as e:
        return False, str(e)


def _banner():
    print("\n" + "=" * 60)
    print("  InsightX — Fintech Intelligence Platform")
    print("  Dependency Checker & Auto-Installer")
    print("=" * 60 + "\n")


def _check_all_packages():
    """Returns list of (import_name, pip_name) tuples for missing required packages."""
    return [
        (imp, pip) for imp, pip in REQUIRED_PACKAGES.items()
        if not _check_import(imp)
    ]


def _check_optional_packages():
    """Returns list of (import_name, pip_name) for missing optional packages."""
    return [
        (imp, pip) for imp, pip in OPTIONAL_PACKAGES.items()
        if not _check_import(imp)
    ]


def _prompt_install(missing_pkgs):
    """Ask the user whether to auto-install. Returns True if yes."""
    print("The following required packages are not installed:\n")
    for imp_name, pip_name in missing_pkgs:
        if imp_name == "tkinter":
            print(f"  ✗  tkinter  (Python built-in — see note below)")
        else:
            print(f"  ✗  {pip_name}")

    tkinter_missing = any(n == "tkinter" for n, _ in missing_pkgs)
    if tkinter_missing:
        print(
            "\n  ⚠  tkinter cannot be pip-installed. Fix it manually:\n"
            "     • Windows : reinstall Python from python.org → tick 'tcl/tk'\n"
            "     • Ubuntu  : sudo apt install python3-tk\n"
            "     • macOS   : brew install python-tk  OR use python.org installer\n"
        )

    pip_installable = [(n, p) for n, p in missing_pkgs if p is not None]
    if not pip_installable:
        print("\nNo packages can be auto-installed. Please fix the above manually.")
        return False

    print(
        f"\nInsightX can auto-install {len(pip_installable)} package(s) for you.\n"
        "Type  yes  to install automatically, or  no  to install yourself.\n"
        "(Tip: just type 'yes' — it usually takes 1-3 minutes)\n"
    )
    try:
        answer = input("  Install missing packages? [yes/no] > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "no"

    return answer in ("yes", "y")


def _run_auto_install(missing_pkgs):
    """Install each pip-installable missing package and return list of failures."""
    pip_installable = [(n, p) for n, p in missing_pkgs if p is not None]
    if not pip_installable:
        return []

    print(f"\n  Installing {len(pip_installable)} package(s)...\n")
    failed = []
    for imp_name, pip_name in pip_installable:
        print(f"  ▶  Installing {pip_name} ...", end="", flush=True)
        ok, out = _pip_install(pip_name)
        if ok:
            print("  ✓")
        else:
            print("  ✗  FAILED")
            last_lines = "\n".join(out.strip().splitlines()[-5:])
            if last_lines:
                print(f"     {last_lines}\n")
            failed.append(pip_name)

    print()
    if failed:
        print(
            f"  ⚠  {len(failed)} package(s) failed: {', '.join(failed)}\n"
            "     Try running as administrator, or install them manually.\n"
        )
    else:
        print("  ✓  All packages installed successfully!\n")

    return failed


def _print_manual_instructions(missing_pkgs):
    pip_installable = [(n, p) for n, p in missing_pkgs if p is not None]
    if not pip_installable:
        return
    print("\n  Manual install commands:\n")
    for _, pip_name in pip_installable:
        print(f"    pip install {pip_name}")
    print()


def main():
    _banner()

    # ── Step 1: Check required packages ──────────────────────────────────────
    missing = _check_all_packages()

    if missing:
        want_install = _prompt_install(missing)
        if want_install:
            _run_auto_install(missing)
            # Re-check
            still_missing = [(n, p) for n, p in _check_all_packages() if p is not None]
            if still_missing:
                print("  ⚠  Some packages still missing. Attempting to launch anyway...\n")
                _print_manual_instructions(still_missing)
                try:
                    input("  Press Enter to continue, or Ctrl+C to exit > ")
                except (EOFError, KeyboardInterrupt):
                    print("\n  Exiting. Install the packages and re-run InsightX.")
                    sys.exit(1)
        else:
            _print_manual_instructions(missing)
            try:
                input("  Press Enter to attempt launch anyway, or Ctrl+C to exit > ")
            except (EOFError, KeyboardInterrupt):
                print("\n  Exiting.")
                sys.exit(1)
    else:
        print("  ✓  All required packages are present.\n")

    # ── Step 2: Optional packages (non-blocking info) ─────────────────────────
    for imp_name, pip_name in _check_optional_packages():
        print(f"  ℹ  Optional: '{pip_name}' not found  →  pip install {pip_name}")
    print()

    # ── Step 3: Silence noisy library output ──────────────────────────────────
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import warnings
    warnings.filterwarnings("ignore")

    import logging
    for lib in ("transformers", "torch", "bitsandbytes", "accelerate",
                "huggingface_hub", "diffusers", "PIL"):
        logging.getLogger(lib).setLevel(logging.ERROR)

    # ── Step 4: Launch InsightX ───────────────────────────────────────────────
    print("  Launching InsightX...\n" + "=" * 60 + "\n")
    try:
        from ui import InsightXApp
        app = InsightXApp()
        app.mainloop()
    except Exception as e:
        print(f"\n  ✗  InsightX crashed on startup: {e}")
        import traceback
        traceback.print_exc()
        try:
            input("\n  Press Enter to exit...")
        except (EOFError, KeyboardInterrupt):
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()

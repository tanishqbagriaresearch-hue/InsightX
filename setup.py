"""
InsightX — First-run setup script
Run this once before launching the app:  python setup.py
"""
import subprocess, sys, os, webbrowser

GGUF_FILENAME = "Phi-3-mini-4k-instruct-q4.gguf"
GGUF_DIR      = os.path.dirname(os.path.abspath(__file__))
GGUF_PATH     = os.path.join(GGUF_DIR, GGUF_FILENAME)
GGUF_DL_URL   = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"

PACKAGES = [
    "pandas",
    "numpy",
    "matplotlib",
    "torch",
    "llama-cpp-python",
    "psutil",
]


def get_python():
    exe = sys.executable
    if exe.lower().endswith("pythonw.exe"):
        alt = exe[:-len("pythonw.exe")] + "python.exe"
        if os.path.isfile(alt):
            return alt
    return exe


def install_packages():
    print("=" * 60)
    print("Step 1/2 — Installing Python dependencies")
    print("=" * 60)
    python = get_python()
    for pkg in PACKAGES:
        print(f"  Installing {pkg}...")
        result = subprocess.run(
            [python, "-m", "pip", "install", pkg],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  WARNING: could not install {pkg}:")
            print(result.stderr[-800:] if result.stderr else result.stdout[-800:])
        else:
            print(f"  OK: {pkg}")
    print("  Done.\n")


def check_gguf():
    if os.path.isfile(GGUF_PATH) and os.path.getsize(GGUF_PATH) > 1_000_000_000:
        print(f"  Model already present: {GGUF_PATH}")
        return

    import tkinter as tk
    from tkinter import messagebox

    print(f"\n  Opening download link in your browser...")
    webbrowser.open(GGUF_DL_URL)

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    msg = (
        f"InsightX cannot run without the AI model file.\n\n"
        f"A download has been started in your browser.\n\n"
        f"File to download:\n"
        f"  {GGUF_FILENAME}  (~2.4 GB)\n\n"
        f"Save it to:\n"
        f"  {GGUF_DIR}\n\n"
        f"Once the download is complete, run:\n"
        f"  python setup.py\n"
        f"then launch with:\n"
        f"  python main.py"
    )
    messagebox.showwarning("Model Required — InsightX", msg)
    root.destroy()

    print("\n  Download the model file and re-run setup.py.")
    sys.exit(0)


if __name__ == "__main__":
    print("\n  InsightX Setup\n")
    install_packages()
    print("=" * 60)
    print("Step 2/2 — Checking for AI model")
    print("=" * 60)
    check_gguf()
    print("\n" + "=" * 60)
    print("  Setup complete! Run the app with:  python main.py")
    print("=" * 60 + "\n")

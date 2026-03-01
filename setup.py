"""
InsightX — First-run setup script
Run this once before launching the app:  python setup.py
"""
import subprocess, sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GGUF_FILENAME = "Phi-3-mini-4k-instruct-q4.gguf"
GGUF_PATH     = os.path.join(SCRIPT_DIR, GGUF_FILENAME)
GGUF_URL      = (
    "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf"
    "/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
)

CSV_FILENAME  = "data.csv"
CSV_PATH      = os.path.join(SCRIPT_DIR, CSV_FILENAME)
CSV_DRIVE_ID  = "1HupU0m9J8Rlxn8BgAqAqdUHwIs9zG1Bz"

PACKAGES = [
    "pandas",
    "numpy",
    "matplotlib",
    "torch",
    "llama-cpp-python",
    "psutil",
    "requests",
    "tqdm",
    "gdown",
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
    print("Step 1/3 — Installing Python dependencies")
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


def download_csv():
    print("=" * 60)
    print("Step 2/3 — Downloading data.csv from Google Drive")
    print("=" * 60)
    import gdown

    # Always re-download — remove old file and replace it
    if os.path.isfile(CSV_PATH):
        print(f"  Removing existing {CSV_FILENAME} and re-downloading...")
        os.remove(CSV_PATH)

    print(f"  Destination: {CSV_PATH}")
    url = f"https://drive.google.com/uc?id={CSV_DRIVE_ID}"
    gdown.download(url, CSV_PATH, quiet=False, fuzzy=True)

    if os.path.isfile(CSV_PATH) and os.path.getsize(CSV_PATH) > 100:
        print(f"  OK: data.csv downloaded ({os.path.getsize(CSV_PATH) / 1e6:.2f} MB)\n")
    else:
        print("\n  ERROR: data.csv download failed or file is empty.")
        print("  Make sure the Google Drive file is shared as 'Anyone with the link'.")
        sys.exit(1)


def download_gguf():
    print("=" * 60)
    print("Step 3/3 — Downloading AI model (~2.4 GB)")
    print("=" * 60)

    if os.path.isfile(GGUF_PATH) and os.path.getsize(GGUF_PATH) > 1_000_000_000:
        print(f"  Model already present: {GGUF_PATH}\n")
        return

    import requests
    from tqdm import tqdm

    print(f"  Downloading {GGUF_FILENAME} ...")
    print(f"  Destination: {GGUF_PATH}")

    response = requests.get(GGUF_URL, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0)) or None

    with open(GGUF_PATH, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        unit_divisor=1024, desc="  Model", ncols=70
    ) as bar:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))

    print(f"  Done — {os.path.getsize(GGUF_PATH) / 1e9:.2f} GB written.\n")


if __name__ == "__main__":
    print("\n  InsightX Setup\n")
    install_packages()
    download_csv()
    download_gguf()
    print("\n" + "=" * 60)
    print("  Setup complete!  Run the app with:  python main.py")
    print("=" * 60 + "\n")

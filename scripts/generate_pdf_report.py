#!/usr/bin/env python3
"""
generate_pdf_report.py

Compiles docs/PROJECT_COMPLETE_REPORT_HE.html into a publication-grade PDF
using headless Google Chrome.
"""

import base64
import os
from pathlib import Path
import subprocess

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
PLOTS_DIR = BASE_DIR / "plots"
TEMPLATE_HTML = DOCS_DIR / "PROJECT_COMPLETE_REPORT_HE.html"
OUTPUT_PDF = DOCS_DIR / "PROJECT_COMPLETE_REPORT_HE.pdf"
ROOT_PDF = BASE_DIR / "PROJECT_COMPLETE_REPORT_HE.pdf"


def get_base64_image(filename):
    path = PLOTS_DIR / filename
    if not path.exists():
        print(f"Warning: Plot {filename} not found.")
        return ""
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def main():
    if not TEMPLATE_HTML.exists():
        print(f"Error: {TEMPLATE_HTML} not found.")
        return

    content = TEMPLATE_HTML.read_text(encoding="utf-8")

    # Replace image placeholders with base64 data URIs
    replacements = {
        "__IMG_TRAJECTORIES__": get_base64_image("trajectories_comparison.png"),
        "__IMG_SCATTER__": get_base64_image("features_scatter.png"),
        "__IMG_VELOCITY__": get_base64_image("velocity_profiles.png"),
        "__IMG_TREE__": get_base64_image("decision_tree_diagram.png"),
        "__IMG_POWER_LAW__": get_base64_image("two_thirds_power_law.png"),
    }

    for placeholder, data_uri in replacements.items():
        content = content.replace(placeholder, data_uri)

    # Write self-contained HTML
    TEMPLATE_HTML.write_text(content, encoding="utf-8")
    print(f"Injected base64 figures into {TEMPLATE_HTML.name}")

    chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not os.path.exists(chrome_path):
        print(f"Error: Chrome binary not found at {chrome_path}")
        return

    print("Compiling to PDF via headless Google Chrome...")
    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--allow-file-access-from-files",
        "--enable-local-file-accesses",
        "--no-sandbox",
        "--virtual-time-budget=6000",
        f"--print-to-pdf={OUTPUT_PDF}",
        str(TEMPLATE_HTML.resolve()),
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        if OUTPUT_PDF.exists() and OUTPUT_PDF.stat().st_size > 1000:
            print(f"SUCCESS: Created PDF at {OUTPUT_PDF} ({OUTPUT_PDF.stat().st_size / 1024:.1f} KB)")
            # Also copy to root for easy access
            ROOT_PDF.write_bytes(OUTPUT_PDF.read_bytes())
            print(f"Copied PDF to root at {ROOT_PDF}")
        else:
            print("Warning: PDF was not generated or empty.")
    except Exception as e:
        print(f"Error executing Chrome print-to-pdf: {e}")


if __name__ == "__main__":
    main()

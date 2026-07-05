import json
import re
from pathlib import Path
import pdfplumber

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def clean_text(text: str) -> str:
    # Remove arXiv watermark lines (backwards text)
    text = re.sub(r"^.*viXra.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^.*\]IS\.sc\[.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^ceD$", "", text, flags=re.MULTILINE)
    # Remove garbled unicode symbols
    text = re.sub(r"[ΓÇáΓêù┬╖\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove page-number-only lines
    lines = [
        line for line in text.split("\n")
        if not re.fullmatch(r"\s*\d+\s*", line)
    ]
    return "\n".join(lines).strip()


def load_pdf(path: Path):
    pages = []
    with pdfplumber.open(path) as pdf:
        num_pages = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return "\n\n".join(pages), num_pages


def process_all():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {RAW_DIR}/")
        return

    print(f"Found {len(pdf_files)} PDF(s). Processing...\n")

    for pdf_path in pdf_files:
        try:
            raw_text, num_pages = load_pdf(pdf_path)
            cleaned = clean_text(raw_text)

            out_name = pdf_path.stem
            txt_out = PROCESSED_DIR / f"{out_name}.txt"
            meta_out = PROCESSED_DIR / f"{out_name}.json"

            txt_out.write_text(cleaned, encoding="utf-8")
            meta_out.write_text(json.dumps({
                "source_file": pdf_path.name,
                "num_pages": num_pages,
                "char_count": len(cleaned),
            }, indent=2))

            print(f"  OK {pdf_path.name} -> {txt_out.name} "
                  f"({num_pages} pages, {len(cleaned)} chars)")

            if len(cleaned) < 200:
                print(f"     WARNING: Very little text - may be scanned PDF")

        except Exception as e:
            print(f"  FAILED {pdf_path.name}: {e}")

    print(f"\nDone. Check data/processed/")


if __name__ == "__main__":
    process_all()
"""Generate PDF study guides for team members using built-in Helvetica."""
from fpdf import FPDF
from pathlib import Path


def clean(text):
    """Remove all non-ASCII and special chars."""
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2192": "->",
        "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"',
        "`": "", "\u00b1": "+/-", "\u00d7": "x",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return "".join(c if ord(c) < 128 else " " for c in text)


def md_to_pdf(md_path, pdf_path):
    content = clean(Path(md_path).read_text(encoding="utf-8"))

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    for line in content.split("\n"):
        line = line.rstrip()

        # Skip table separator lines like |---|---|
        if line.startswith("|") and set(line.replace("|", "").replace("-", "").replace(" ", "").replace(":", "")) == set():
            continue

        if not line.strip():
            pdf.ln(3)
            continue

        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 15)
            pdf.multi_cell(180, 8, line[2:])
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.ln(2)
            pdf.multi_cell(180, 7, line[3:])
            pdf.ln(1)
        elif line.startswith("### "):
            pdf.set_font("Helvetica", "B", 10)
            pdf.ln(1)
            pdf.multi_cell(180, 6, line[4:])
            pdf.ln(1)
        elif line.startswith("|"):
            pdf.set_font("Helvetica", "", 9)
            cells = [c.strip() for c in line.split("|") if c.strip()]
            row = " | ".join(cells)
            pdf.multi_cell(180, 5, row)
        elif line.startswith("---"):
            pdf.ln(1)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())
            pdf.ln(2)
        elif line.startswith("**"):
            pdf.set_font("Helvetica", "B", 10)
            cleaned = line.replace("**", "").replace("*", "")
            pdf.multi_cell(180, 5.5, cleaned)
        elif line.startswith("- "):
            pdf.set_font("Helvetica", "", 10)
            # Replace bullet dash with bullet char
            pdf.multi_cell(180, 5.5, "  " + line[2:])
        else:
            pdf.set_font("Helvetica", "", 10)
            cleaned = line.replace("*", "")
            pdf.multi_cell(180, 5.5, cleaned)

    pdf.output(pdf_path)
    print(f"Created: {pdf_path}")


if __name__ == "__main__":
    pairs = [
        ("paper/guide_fonseka.md", "paper/guide_Fonseka.pdf"),
        ("paper/guide_manduli.md", "paper/guide_Manduli.pdf"),
        ("paper/guide_bandara.md", "paper/guide_Bandara.pdf"),
    ]
    for md, out in pairs:
        md_to_pdf(md, out)
    print("\nAll study guides generated!")


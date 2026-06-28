"""Generate PDF versions of video narration scripts."""
from fpdf import FPDF
from pathlib import Path
import re

files = [
    ("paper/video_perera.md", "paper/video_script_Perera.pdf"),
    ("paper/video_fonseka.md", "paper/video_script_Fonseka.pdf"),
    ("paper/video_manduli.md", "paper/video_script_Manduli.pdf"),
    ("paper/video_bandara.md", "paper/video_script_Bandara.pdf"),
]

for md_path, pdf_path in files:
    content = Path(md_path).read_text(encoding="utf-8")

    # Clean all special characters
    content = content.replace("\u2014", "-").replace("\u2013", "-")
    content = content.replace("\u2018", "'").replace("\u2019", "'")
    content = content.replace("\u201c", '"').replace("\u201d", '"')
    content = content.replace("`", "")
    content = content.replace("\u2192", "->")
    content = content.replace("\u2190", "<-")
    content = content.replace("\u00b1", "+/-")
    # Strip any remaining non-ASCII
    content = content.encode("ascii", "replace").decode("ascii")

    pdf = FPDF()
    pdf.add_font("Arial", "", "C:/Windows/Fonts/arial.ttf")
    pdf.add_font("Arial", "B", "C:/Windows/Fonts/arialbd.ttf")
    pdf.add_font("Arial", "I", "C:/Windows/Fonts/ariali.ttf")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    for line in content.split("\n"):
        line = line.rstrip()

        # Skip table separators
        if re.match(r"^\|[-\s|]+\|$", line):
            continue

        if line.startswith("# "):
            pdf.set_font("Arial", "B", 16)
            pdf.multi_cell(180, 8, line[2:])
            pdf.ln(2)
        elif line.startswith("## "):
            pdf.set_font("Arial", "B", 12)
            pdf.multi_cell(180, 7, line[3:])
            pdf.ln(1)
        elif line.startswith("### "):
            pdf.set_font("Arial", "B", 11)
            pdf.multi_cell(180, 6, line[4:])
            pdf.ln(1)
        elif line.startswith("**"):
            pdf.set_font("Arial", "B", 10)
            clean = line.replace("**", "").replace("*", "")
            pdf.multi_cell(180, 5.5, clean)
            pdf.ln(1)
        elif line.startswith("*(") or line.startswith("*["):
            pdf.set_font("Arial", "I", 9)
            clean = line.replace("*", "").replace("(", "").replace(")", "")
            pdf.multi_cell(180, 5, clean)
            pdf.ln(1)
        elif line.startswith("|"):
            pdf.set_font("Arial", "", 9)
            cells = [c.strip() for c in line.split("|") if c.strip()]
            clean = "  |  ".join(cells)
            pdf.cell(0, 4.5, clean, new_x="LMARGIN", new_y="NEXT")
        elif line.startswith("---"):
            pdf.ln(1)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())
            pdf.ln(2)
        elif line.strip() == "":
            pdf.ln(2)
        else:
            pdf.set_font("Arial", "", 10)
            clean = line.replace("*", "")
            pdf.multi_cell(180, 5.5, clean)

    pdf.output(pdf_path)
    print(f"Created: {pdf_path}")

print("\nAll 4 PDFs generated!")


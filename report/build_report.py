"""Build REPORT.pdf from report/REPORT.md (needs pandoc and xelatex on PATH).

`<!-- include: path -->` lines are replaced by that file's body, so the Task 1 write-up has one source
(task1/WRITEUP.md) and the report cannot drift from it. Paths in the report are relative to the repo root.
"""
import re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "report" / "REPORT.md", ROOT / "REPORT.pdf"
MAX_PAGES = 6  # brief: at most 6 pages covering both tasks


def body(path):
    # drop the file's own H1 title and the source line under it; the report supplies its own heading level
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    while lines and (lines[0].startswith("# ") or lines[0].startswith("Numbers come from") or not lines[0].strip()):
        lines.pop(0)
    # demote "## (a) ..." one level so it sits under the report's Write-up heading
    return "## Write-up\n\n" + re.sub(r"^## ", "### ", "\n".join(lines), flags=re.M)


def cropped(path, tmpdir):
    # analysis figures carry wide white margins; trimming them (pixels untouched) saves page space under the 6-page cap
    from PIL import Image, ImageChops
    im = Image.open(ROOT / path).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    out = tmpdir / Path(path).name
    (im.crop(box) if box else im).save(out)
    return out.relative_to(ROOT).as_posix()


def main():
    for tool in ("pandoc", "xelatex"):
        if not shutil.which(tool):
            # not an error: every number is already in outputs/, the PDF only typesets them
            print(f"{tool} not found on PATH; REPORT.pdf not rebuilt (committed copy kept)")
            return
    tmpdir = ROOT / "report" / "_build"
    tmpdir.mkdir(exist_ok=True)
    md = re.sub(r"<!-- include: (\S+) -->", lambda m: body(m.group(1)), SRC.read_text(encoding="utf-8"))
    md = re.sub(r"\]\(([^)]+\.png)\)", lambda m: f"]({cropped(m.group(1), tmpdir)})", md)
    tmp = tmpdir / "_expanded.md"
    tmp.write_text(md, encoding="utf-8")
    # 10 pt, 1.8 cm margins: the densest layout that stays readable inside the 6-page limit
    subprocess.run(["pandoc", str(tmp), "-o", str(OUT), "--pdf-engine=xelatex", f"--resource-path={ROOT}",
                    "-V", "geometry:margin=1.8cm", "-V", "fontsize=10pt", "-V", "mainfont=Calibri",
                    "-V", "colorlinks=true"], check=True, cwd=ROOT)
    shutil.rmtree(tmpdir)
    try:
        import fitz
        n = len(fitz.open(OUT))
        print(f"REPORT.pdf: {n} pages" + (f"  (OVER the {MAX_PAGES}-page limit)" if n > MAX_PAGES else ""))
    except ImportError:
        print("REPORT.pdf written (page count not checked: PyMuPDF missing)")


if __name__ == "__main__":
    main()

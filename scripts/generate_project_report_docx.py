"""Generate the NIT Patna formatted Machine Learning Project Report (.docx) in docx/."""

import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


NAVY = RGBColor(0x1B, 0x36, 0x5D)
DARK_TEXT = RGBColor(0x1F, 0x24, 0x2B)
MUTED_GRAY = RGBColor(0x55, 0x5E, 0x68)


def create_architecture_diagram(out_path: Path) -> Path:
    """Render a crisp publication-grade system architecture diagram PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, subtitle: str, fc: str, ec: str = "#1b365d") -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.5,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h * 0.63,
            title,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#102030",
        )
        if subtitle:
            ax.text(
                x + w / 2,
                y + h * 0.30,
                subtitle,
                ha="center",
                va="center",
                fontsize=7.8,
                color="#2c3e50",
            )

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#1b365d", lw=1.4, shrinkA=2, shrinkB=2),
        )

    # Top: Dataset & Leakage-Free Split
    box(0.4, 4.9, 3.0, 0.95, "PlantVillage Raw RGB", "54,305 images | 38 classes | 14 crops", "#e8f1fa")
    box(4.0, 4.9, 3.1, 0.95, "Audit & MD5 Deduplication", "0 corrupt | 21 exact dupes removed\n54,284 unique usable images", "#e8f1fa")
    box(7.6, 4.9, 3.0, 0.95, "Leaf-Grouped Stratified Split", "StratifiedGroupKFold (20,015 leaves)\nTrain 38,741 | Val 7,718 | Test 7,825", "#dceefb")

    arrow(3.45, 5.38, 3.95, 5.38)
    arrow(7.15, 5.38, 7.55, 5.38)

    # Left Branch: Handcrafted Image Features
    box(0.5, 3.1, 4.4, 1.15, "Pipeline A: Handcrafted Image Features (114 dims)", "128x128 RGB/HSV/LAB stats+hists (66) | ExG/GRVI/Lesion (8)\nGLCM + Uniform LBP Texture (30) | Hu Moments + Edges (10)", "#edf7ed", "#2e7d32")
    box(0.5, 1.65, 2.05, 1.0, "Classical ML", "Logistic Regression\nRandom Forest (200)", "#f4faf4", "#2e7d32")
    box(2.85, 1.65, 2.05, 1.0, "Feature-MLP (NN)", "114 -> 256 -> 128 -> 64\nStandardScaler + AdamW", "#e3f3e3", "#2e7d32")

    # Right Branch: End-to-End CNNs
    box(6.0, 3.1, 4.5, 1.15, "Pipeline B: Deep CNNs (224x224 RGB)", "v2 Augmentations + ImageNet Norm + Sqrt Class Weights\nCUDA AMP (float16) | AdamW + CosineAnnealingLR", "#fff4e5", "#b26a00")
    box(6.0, 1.65, 2.1, 1.0, "Custom CNN (Scratch)", "4 Conv Blocks + GAP\n1.21M params (Bin & 38-Cls)", "#fff8ee", "#b26a00")
    box(8.4, 1.65, 2.1, 1.0, "MobileNetV2 Transfer", "IMAGENET1K_V2 Backbone\n2.27M params + Grad-CAM", "#ffeccc", "#b26a00")

    # Routing arrows from split
    arrow(8.4, 4.85, 2.7, 4.30)
    arrow(9.1, 4.85, 8.25, 4.30)

    arrow(1.9, 3.05, 1.5, 2.70)
    arrow(3.5, 3.05, 3.85, 2.70)
    arrow(7.4, 3.05, 7.05, 2.70)
    arrow(9.1, 3.05, 9.45, 2.70)

    # Bottom: Evaluation, Rollup & App
    box(1.2, 0.25, 8.6, 0.95, "Unified Evaluation, Binary Rollup & Streamlit Diagnostic App", "38-Class Disease ID + Healthy vs. Diseased Rollup | ROC-AUC & Confusion Matrices | Grad-CAM Heatmaps & Feature Inspector", "#f0edf8", "#4a307d")

    arrow(1.5, 1.60, 2.8, 1.25)
    arrow(3.85, 1.60, 4.3, 1.25)
    arrow(7.05, 1.60, 6.7, 1.25)
    arrow(9.45, 1.60, 8.2, 1.25)

    plt.title("System Architecture: Plant Disease Classification Using Image Features & Neural Networks", fontsize=11.5, fontweight="bold", color="#102030", pad=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def set_cell_shading(cell, fill_hex: str) -> None:
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def set_cell_margins(cell, top: int = 80, bottom: int = 80, left: int = 120, right: int = 120) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for side, val in (("top", top), ("bottom", bottom), ("left", left), ("right", right)):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def set_repeat_table_header(row) -> None:
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:tblHeader"))


def add_page_number_to_run(run) -> None:
    fldChar1 = parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls("w"))
    instrText = parse_xml(r'<w:instrText %s xml:space="preserve"> PAGE </w:instrText>' % nsdecls("w"))
    fldChar2 = parse_xml(r'<w:fldChar %s w:fldCharType="separate"/>' % nsdecls("w"))
    fldChar3 = parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls("w"))
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)


def add_toc_field(doc: Document) -> None:
    p = doc.add_paragraph()
    run = p.add_run()
    fldChar = parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls("w"))
    instrText = parse_xml(r'<w:instrText %s xml:space="preserve"> TOC \o "1-3" \h \z \u </w:instrText>' % nsdecls("w"))
    fldChar2 = parse_xml(r'<w:fldChar %s w:fldCharType="separate"/>' % nsdecls("w"))
    fldChar3 = parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls("w"))
    run._r.append(fldChar)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)


def add_styled_table(
    doc: Document,
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    col_widths: Sequence[float] | None = None,
    font_size: int = 9,
) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr_row = table.rows[0]
    set_repeat_table_header(hdr_row)
    for idx, text in enumerate(headers):
        cell = hdr_row.cells[idx]
        set_cell_shading(cell, "1B365D")
        set_cell_margins(cell, top=90, bottom=90, left=100, right=100)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(str(text))
        run.bold = True
        run.font.name = "Times New Roman"
        run.font.size = Pt(font_size)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        if col_widths:
            cell.width = Inches(col_widths[idx])

    for r_idx, row_vals in enumerate(rows):
        row = table.add_row()
        bg = "F4F7FA" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_vals):
            cell = row.cells[c_idx]
            set_cell_shading(cell, bg)
            set_cell_margins(cell, top=65, bottom=65, left=100, right=100)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(str(val))
            run.font.name = "Times New Roman"
            run.font.size = Pt(font_size)
            run.font.color.rgb = DARK_TEXT
            if col_widths:
                cell.width = Inches(col_widths[c_idx])

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(4)


def add_body_p(doc: Document, text: str, bold_prefix: str | None = None, space_after: int = 6) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(space_after)
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.bold = True
        rb.font.name = "Times New Roman"
        rb.font.size = Pt(11.5)
        rb.font.color.rgb = DARK_TEXT
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(11.5)
    r.font.color.rgb = DARK_TEXT


def add_bullet(doc: Document, text: str, bold_prefix: str | None = None) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(3)
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.bold = True
        rb.font.name = "Times New Roman"
        rb.font.size = Pt(11)
        rb.font.color.rgb = DARK_TEXT
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    r.font.color.rgb = DARK_TEXT


def add_code_block(doc: Document, code_text: str) -> None:
    tbl = doc.add_table(rows=1, cols=1)
    tbl.style = "Table Grid"
    cell = tbl.rows[0].cells[0]
    set_cell_shading(cell, "F5F7F9")
    set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.05
    r = p.add_run(code_text.strip())
    r.font.name = "Consolas"
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(4)


def add_figure(doc: Document, img_path: Path, caption: str, width_in: float = 5.9) -> None:
    if not img_path.exists():
        add_body_p(doc, f"[Missing figure: {img_path}]")
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    p_img.add_run().add_picture(str(img_path), width=Inches(width_in))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_after = Pt(8)
    rc = p_cap.add_run(caption)
    rc.italic = True
    rc.font.name = "Times New Roman"
    rc.font.size = Pt(10)
    rc.font.color.rgb = MUTED_GRAY


def add_chapter_heading(doc: Document, title: str, page_break: bool = True) -> None:
    if page_break:
        doc.add_page_break()
    h = doc.add_heading(title, level=1)
    h.alignment = WD_ALIGN_PARAGRAPH.LEFT
    h.paragraph_format.space_before = Pt(6)
    h.paragraph_format.space_after = Pt(10)
    for r in h.runs:
        r.font.name = "Times New Roman"
        r.font.size = Pt(16)
        r.bold = True
        r.font.color.rgb = NAVY


def add_sec_heading(doc: Document, title: str) -> None:
    h = doc.add_heading(title, level=2)
    h.paragraph_format.space_before = Pt(10)
    h.paragraph_format.space_after = Pt(4)
    for r in h.runs:
        r.font.name = "Times New Roman"
        r.font.size = Pt(13)
        r.bold = True
        r.font.color.rgb = NAVY


def add_subsec_heading(doc: Document, title: str) -> None:
    h = doc.add_heading(title, level=3)
    h.paragraph_format.space_before = Pt(7)
    h.paragraph_format.space_after = Pt(3)
    for r in h.runs:
        r.font.name = "Times New Roman"
        r.font.size = Pt(11.5)
        r.bold = True
        r.font.color.rgb = DARK_TEXT


def add_centered_line(
    doc: Document,
    text: str,
    size: int = 12,
    bold: bool = False,
    italic: bool = False,
    color: RGBColor = DARK_TEXT,
    space_after: int = 6,
) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    r.font.color.rgb = color


def build_report() -> Path:
    metrics_dir = PROJECT_ROOT / "outputs" / "metrics"
    figures_dir = PROJECT_ROOT / "outputs" / "figures"
    docx_dir = PROJECT_ROOT / "docx"
    docx_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary = json.loads((metrics_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    bench_df = pd.read_csv(metrics_dir / "benchmark_summary.csv")
    report_38_df = pd.read_csv(metrics_dir / "transfer_cnn_38class_report.csv", index_col=0)
    error_analysis = json.loads((metrics_dir / "error_analysis_transfer_cnn.json").read_text(encoding="utf-8"))
    splits_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "dataset_splits.csv")

    arch_png = create_architecture_diagram(figures_dir / "system_architecture_diagram.png")

    project_title = "Plant Disease Classification Using Image Features and Neural Networks"

    doc = Document()
    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.85)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(0.9)
        section.different_first_page_header_footer = True

        hdr = section.header.paragraphs[0]
        hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hr = hdr.add_run(project_title)
        hr.font.name = "Times New Roman"
        hr.font.size = Pt(9)
        hr.italic = True
        hr.font.color.rgb = MUTED_GRAY

        ftr = section.footer.paragraphs[0]
        ftr.alignment = WD_ALIGN_PARAGRAPH.CENTER
        fr1 = ftr.add_run("Department of CSE, NIT Patna  |  Page ")
        fr1.font.name = "Times New Roman"
        fr1.font.size = Pt(9)
        fr1.font.color.rgb = MUTED_GRAY
        fr_num = ftr.add_run()
        fr_num.font.name = "Times New Roman"
        fr_num.font.size = Pt(9)
        fr_num.font.color.rgb = MUTED_GRAY
        add_page_number_to_run(fr_num)

    # Set default Normal style
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11.5)
    normal.font.color.rgb = DARK_TEXT

    # =========================================================================
    # PAGE 1: TITLE PAGE
    # =========================================================================
    add_centered_line(doc, "MACHINE LEARNING PROJECT REPORT", size=16, bold=True, color=NAVY, space_after=18)
    add_centered_line(doc, project_title.upper(), size=18, bold=True, color=DARK_TEXT, space_after=18)
    add_centered_line(
        doc,
        "Submitted in partial fulfillment of the requirements for the award of the degree of",
        size=11.5,
        italic=True,
        space_after=8,
    )
    add_centered_line(
        doc,
        "Bachelor of Technology (B.Tech.)\nin\nComputer Science and Engineering",
        size=13,
        bold=True,
        color=NAVY,
        space_after=22,
    )

    add_centered_line(doc, "Submitted By:", size=12, bold=True, space_after=8)
    add_centered_line(doc, "Name: [Student Name 1]   |   Roll No.: [Roll Number 1]", size=11.5, space_after=4)
    add_centered_line(doc, "Name: [Student Name 2]   |   Roll No.: [Roll Number 2]", size=11.5, space_after=4)
    add_centered_line(doc, "Name: [Student Name 3]   |   Roll No.: [Roll Number 3]", size=11.5, space_after=22)

    add_centered_line(doc, "Under the Guidance of", size=12, bold=True, space_after=6)
    add_centered_line(doc, "[Name of Project Supervisor]", size=12.5, bold=True, color=NAVY, space_after=2)
    add_centered_line(doc, "[Designation]", size=11.5, italic=True, space_after=28)

    add_centered_line(doc, "DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING", size=12.5, bold=True, color=NAVY, space_after=4)
    add_centered_line(doc, "NATIONAL INSTITUTE OF TECHNOLOGY PATNA", size=14, bold=True, color=NAVY, space_after=4)
    add_centered_line(doc, "Patna, Bihar – 800005", size=11.5, space_after=16)
    add_centered_line(doc, "Academic Session: [2026–2027]", size=12, bold=True, space_after=0)

    # =========================================================================
    # PAGE 2: CERTIFICATE
    # =========================================================================
    add_chapter_heading(doc, "CERTIFICATE", page_break=True)
    add_body_p(
        doc,
        f'This is to certify that the project report entitled "{project_title}" submitted by '
        "[Student Name(s)], bearing Roll Number(s) [Roll Number(s)], in partial fulfillment of the "
        "requirements for the award of the degree of Bachelor of Technology in Computer Science and "
        "Engineering at the National Institute of Technology Patna, is a bona fide record of the work "
        "carried out by them under my supervision.",
        space_after=12,
    )
    add_body_p(
        doc,
        "The work presented in this report has not been submitted elsewhere for the award of any other "
        "degree or diploma.",
        space_after=36,
    )

    sig_tbl = doc.add_table(rows=1, cols=2)
    sig_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_left, c_right = sig_tbl.rows[0].cells
    c_left.width = Inches(3.2)
    c_right.width = Inches(3.2)
    p_l = c_left.paragraphs[0]
    p_l.add_run("Project Supervisor\n\nName: [Supervisor Name]\nDesignation: [Designation]\nDepartment of CSE, NIT Patna").font.size = Pt(11)
    p_r = c_right.paragraphs[0]
    p_r.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_r.add_run("Head of Department\n\n\nDepartment of CSE, NIT Patna").font.size = Pt(11)

    p_dp = doc.add_paragraph()
    p_dp.paragraph_format.space_before = Pt(32)
    p_dp.add_run("Date: __________________\nPlace: Patna").font.size = Pt(11)

    # =========================================================================
    # PAGE 3: DECLARATION
    # =========================================================================
    add_chapter_heading(doc, "DECLARATION", page_break=True)
    add_body_p(
        doc,
        f'We hereby declare that the project entitled "{project_title}" is our original work carried out '
        "under the supervision of [Supervisor Name], Department of Computer Science and Engineering, "
        "National Institute of Technology Patna.",
        space_after=12,
    )
    add_body_p(
        doc,
        "We further declare that this work has not been submitted to any other institution or university "
        "for the award of any degree or diploma. All sources of information and references used in this "
        "project have been duly acknowledged.",
        space_after=28,
    )
    add_body_p(doc, "Student Name(s): [Student Name(s)]", space_after=6)
    add_body_p(doc, "Roll Number(s): [Roll Number(s)]", space_after=6)
    add_body_p(doc, "Signature(s): ____________________________________", space_after=6)
    add_body_p(doc, "Date: __________________", space_after=0)

    # =========================================================================
    # PAGE 4: ACKNOWLEDGEMENT
    # =========================================================================
    add_chapter_heading(doc, "ACKNOWLEDGEMENT", page_break=True)
    add_body_p(
        doc,
        "We express our sincere gratitude to our project supervisor, [Supervisor Name], for valuable "
        "guidance, encouragement, and continuous support throughout the project.",
        space_after=10,
    )
    add_body_p(
        doc,
        "We would also like to thank the Head of the Department, Department of Computer Science and "
        "Engineering, National Institute of Technology Patna, for providing the necessary facilities "
        "and academic support.",
        space_after=10,
    )
    add_body_p(
        doc,
        "Finally, we extend our gratitude to our faculty members, friends, and family for their "
        "encouragement and support.",
        space_after=0,
    )

    # =========================================================================
    # PAGE 5: ABSTRACT
    # =========================================================================
    add_chapter_heading(doc, "ABSTRACT", page_break=True)
    add_body_p(
        doc,
        "Plant diseases pose a major threat to global food security, crop yield stability, and sustainable "
        "agriculture. Manual visual inspection of foliage by agricultural experts is slow, subjective, and "
        "unscalable across smallholder farms, motivating automated computer-vision-based diagnostic tools. "
        "This project addresses automated plant disease classification—categorizing plant leaf images into "
        "Healthy and Diseased classes alongside fine-grained crop–disease identification—using both handcrafted "
        "image features and neural networks. Experiments are conducted on the open-access PlantVillage dataset "
        "(spMohanty/PlantVillage-Dataset, raw RGB color subset), comprising 54,305 scanned leaf images across "
        "14 crop species and 38 classes (12 healthy, 26 diseased). After integrity verification and removal of "
        "21 exact MD5 duplicate files, 54,284 unique images are partitioned via leakage-free physical leaf-grouped "
        "stratified splitting (StratifiedGroupKFold over 20,015 leaf groups) into Train (38,741), Validation (7,718), "
        "and Test (7,825) sets. We build and compare two complementary pipelines: (1) a 114-dimensional handcrafted "
        "image feature extractor (RGB, HSV, and CIELAB color moments/histograms, Excess Green and chlorosis/lesion "
        "indices, rotation-invariant GLCM and uniform LBP texture descriptors, Hu moments, and Sobel/Canny edges) "
        "evaluated with Logistic Regression, Random Forest, and a PyTorch Feature-MLP neural network; and (2) end-to-end "
        "deep learning via a 4-stage Custom CNN built from scratch and a fine-tuned MobileNetV2 Transfer Learning CNN "
        "with Grad-CAM visual explainability. On the unseen 7,825-leaf test set, the Feature-MLP achieves 98.94% binary "
        "accuracy (0.9993 ROC-AUC) and 96.10% 38-class accuracy; the Custom CNN achieves 97.65% 38-class and 99.71% "
        "rolled-up binary accuracy; and MobileNetV2 reaches 99.64% 38-class accuracy (99.49% Macro-F1) and 99.95% "
        "rolled-up Healthy vs. Diseased accuracy. The complete system is deployed in an interactive Streamlit web "
        "application, demonstrating that multi-class neural networks combined with interpretable image features "
        "deliver accurate, transparent plant pathology screening.",
        space_after=12,
    )
    add_body_p(
        doc,
        "Machine Learning, Plant Disease Classification, PlantVillage, Handcrafted Image Features, "
        "Multi-Layer Perceptron, Convolutional Neural Network, MobileNetV2, Grad-CAM.",
        bold_prefix="Keywords: ",
    )

    # =========================================================================
    # PAGE 6: TABLE OF CONTENTS
    # =========================================================================
    add_chapter_heading(doc, "TABLE OF CONTENTS", page_break=True)
    add_body_p(
        doc,
        "Note: Right-click the field below in Microsoft Word and select 'Update Field' to refresh exact page numbers if edited.",
        space_after=6,
    )
    add_toc_field(doc)
    add_styled_table(
        doc,
        headers=["Section", "Title", "Contents Summary"],
        rows=[
            ["Front Matter", "Certificate, Declaration, Acknowledgement, Abstract", "Formal approvals and project summary"],
            ["Chapter 1", "Introduction", "Background, problem statement, motivation, objectives, scope, report organization"],
            ["Chapter 2", "Literature Review", "Review of 6 peer-reviewed studies, comparative table, research gap, proposed solution"],
            ["Chapter 3", "Dataset Description and Preprocessing", "PlantVillage audit, MD5 deduplication, EDA, 114 features, leaf-grouped splits"],
            ["Chapter 4", "Proposed Methodology", "System architecture, formulations, Feature-MLP, Custom CNN, MobileNetV2, Grad-CAM"],
            ["Chapter 5", "Experimental Setup and Implementation", "Hardware/software stack, CUDA AMP training, learning curves, test protocol"],
            ["Chapter 6", "Results and Performance Evaluation", "Binary & 38-class benchmarks, ROC/confusion/Grad-CAM analysis, error study"],
            ["Chapter 7", "Conclusion and Future Scope", "Summary of achievements, domain limitations, and deployment roadmap"],
            ["References", "References", "IEEE-formatted citations of datasets, architectures, and research papers"],
            ["Appendices", "Appendix A, B, C", "Core source code, full 38-class per-class report, sample outputs & UI tests"],
        ],
        col_widths=[1.2, 2.3, 3.0],
        font_size=9.5,
    )

    # =========================================================================
    # CHAPTER 1: INTRODUCTION
    # =========================================================================
    add_chapter_heading(doc, "CHAPTER 1: INTRODUCTION", page_break=True)

    add_sec_heading(doc, "1.1 Background")
    add_body_p(
        doc,
        "Agricultural productivity relies heavily on timely monitoring of crop health. Plant foliar diseases "
        "caused by fungi, bacteria, oomycetes, viruses, and pests account for substantial annual crop losses "
        "worldwide. Traditionally, disease diagnosis depends on visual scouting by trained agronomists or "
        "laboratory pathology assays—both of which are expensive, time-consuming, and often unavailable in "
        "rural farming regions. Advances in computer vision, digital leaf imaging, and neural networks enable "
        "automated, smartphone-accessible screening systems capable of identifying leaf pathology from raw RGB "
        "photographs within milliseconds."
    )

    add_sec_heading(doc, "1.2 Problem Statement")
    add_body_p(
        doc,
        "Given an RGB photograph of a plant leaf, the objective is to classify the leaf image into Healthy "
        "and Diseased categories—and further identify the exact crop species and disease class among 38 "
        "PlantVillage categories—using both extracted image features and neural networks."
    )

    add_sec_heading(doc, "1.3 Motivation")
    add_body_p(
        doc,
        "Early detection enables targeted intervention (such as localized pruning or precision fungicide "
        "application) before an outbreak spreads across an entire field, lowering chemical input costs and "
        "environmental runoff. From a machine learning perspective, plant leaf images exhibit rich visual "
        "signals—chlorophyll loss (yellowing/chlorosis), necrotic spots, powdery coatings, and rust "
        "pustules—making the domain ideal for comparing classical computer vision feature engineering against "
        "end-to-end convolutional neural networks."
    )

    add_sec_heading(doc, "1.4 Objectives")
    add_body_p(doc, "The major objectives of this project are:")
    for obj in [
        "To identify and formulate plant leaf disease classification as both a Binary (Healthy vs. Diseased) and 38-class fine-grained problem.",
        "To collect, audit, and analyze the PlantVillage RGB dataset while eliminating duplicate files and physical leaf-level data leakage across splits.",
        "To perform domain-grounded image feature engineering extracting 114 color, vegetation-index, texture, and shape/edge descriptors.",
        "To implement and train classical ML baselines (Logistic Regression, Random Forest) alongside neural networks (Feature-MLP, Custom CNN from scratch, and MobileNetV2 Transfer Learning CNN).",
        "To evaluate and compare all models rigorously using Accuracy, Macro Precision, Macro Recall, Macro F1-score, ROC-AUC, and confusion matrices.",
        "To develop an explainable predictive application with Grad-CAM visualizations and an interactive Streamlit web interface.",
    ]:
        add_bullet(doc, obj)

    add_sec_heading(doc, "1.5 Scope of the Project")
    add_body_p(
        doc,
        "The project covers 54,284 unique color images across 14 crop species (Apple, Blueberry, Cherry, "
        "Corn/Maize, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, "
        "and Tomato) and 38 disease/healthy classes from PlantVillage. It delivers reproducible CLI pipelines, "
        "11 automated pytest unit tests, trained checkpoints, publication figures, and a local Streamlit app. "
        "Scope limitations include PlantVillage's controlled laboratory background setting and single-leaf, "
        "single-label image assumptions, which are analyzed critically in Chapters 6 and 7."
    )

    add_sec_heading(doc, "1.6 Organization of the Report")
    add_body_p(
        doc,
        "The remainder of this report is organized as follows: Chapter 2 reviews landmark research in image-based "
        "plant disease recognition and identifies key methodological gaps. Chapter 3 presents the PlantVillage "
        "dataset audit, EDA, 114 handcrafted image features, and leakage-free leaf-grouped splits. Chapter 4 "
        "details the system architecture, mathematical formulations, and neural network pipelines. Chapter 5 "
        "covers hardware/software environment, training configurations, and learning dynamics. Chapter 6 "
        "analyzes quantitative results, confusion matrices, Grad-CAM explainability, and failure modes. "
        "Chapter 7 concludes the study and outlines future field-deployment improvements."
    )

    # =========================================================================
    # CHAPTER 2: LITERATURE REVIEW
    # =========================================================================
    add_chapter_heading(doc, "CHAPTER 2: LITERATURE REVIEW", page_break=True)

    add_sec_heading(doc, "2.1 Introduction")
    add_body_p(
        doc,
        "Automated plant pathology evolved from early threshold-based leaf segmentation and handcrafted "
        "color/texture descriptors (such as GLCM and color histograms paired with SVMs or MLPs) to deep "
        "Convolutional Neural Networks (CNNs) following the release of large-scale curated datasets such as "
        "PlantVillage (2015–2016). Subsequent studies explored deeper transfer-learning backbones, saliency "
        "visualization for trustworthiness, and domain generalization to cluttered real-world field images."
    )

    add_sec_heading(doc, "2.2 Review of Existing Research Works")
    papers = [
        (
            "1. Mohanty et al. (2016) — Using Deep Learning for Image-Based Plant Disease Detection",
            "Mohanty, Hughes, and Salathé (Frontiers in Plant Science, 2016) formulated large-scale image-based "
            "diagnosis across 14 crops and 26 diseases (38 classes) using 54,306 PlantVillage images. Training "
            "AlexNet and GoogLeNet across color, grayscale, and leaf-segmented subsets with transfer learning "
            "and from-scratch baselines, they achieved up to 99.34% accuracy (GoogLeNet). Limitations included "
            "sharp accuracy drops (~31%) when tested on non-PlantVillage real-world images due to uniform lab backgrounds."
        ),
        (
            "2. Sladojevic et al. (2016) — Deep Neural Networks Based Recognition of Plant Diseases by Leaf Image Classification",
            "Sladojevic et al. (Computational Intelligence and Neuroscience, 2016) addressed recognition of 13 "
            "plant diseases plus healthy leaves using a custom internet-collected dataset augmented via geometric "
            "transformations. Using CaffeNet (an AlexNet variant), the framework achieved 96.3% overall precision "
            "(91%–98% per class). Its limitation was a relatively small original sample pool prior to augmentation "
            "and reliance on older high-parameter AlexNet-style fully connected layers."
        ),
        (
            "3. Brahimi et al. (2017) — Deep Learning for Tomato Diseases: Classification and Symptoms Visualization",
            "Brahimi, Boukhalfa, and Moussaoui (Applied Artificial Intelligence, 2017) focused on classifying "
            "9 tomato leaf diseases using 14,828 tomato images from PlantVillage. Comparing shallow handcrafted "
            "classifiers against fine-tuned AlexNet and GoogLeNet, they achieved 99.18% accuracy and introduced "
            "occlusion/saliency visualizations to localize symptom regions. Limitations included restriction to a "
            "single crop species (Tomato) and high computational cost of occlusion maps compared to Grad-CAM."
        ),
        (
            "4. Ferentinos (2018) — Deep Learning Models for Plant Disease Detection and Diagnosis",
            "Ferentinos (Computers and Electronics in Agriculture, 2018) evaluated five CNN architectures (AlexNet, "
            "AlexNetOWTBn, GoogLeNet, OverFeat, and VGG) across 87,848 images covering 58 plant–disease classes "
            "(extended PlantVillage corpus). VGG achieved the highest test success rate of 99.53% (99.48% reported "
            "in baseline summaries). However, VGG contains >130M parameters, making mobile edge deployment difficult, "
            "and performance degraded when training on lab images and testing on field photos."
        ),
        (
            "5. Too et al. (2019) — A Comparative Study of Fine-Tuning Deep Learning Models for Plant Disease Identification",
            "Too et al. (Computers and Electronics in Agriculture, 2019) benchmarked fine-tuned VGG16, InceptionV4, "
            "ResNet-50/101/152, and DenseNet-121/169/201 on the 38-class PlantVillage dataset. DenseNet-121 "
            "achieved 99.75% accuracy with far fewer parameters than VGG16 due to dense feature reuse. Limitations "
            "included absence of explicit leaf-group leakage auditing in random splits and lack of handcrafted feature baselines."
        ),
        (
            "6. Singh et al. (2020) — PlantDoc: A Dataset for Visual Plant Disease Detection",
            "Singh et al. (ACM IKDD CoDS-COMAD, 2020) highlighted the domain shift problem by introducing PlantDoc, "
            "a 2,598-image dataset of natural, cluttered-background field photographs across 13 plant species and "
            "17 disease classes. Object detection and classification models (Faster R-CNN, SSD, InceptionResNetV2) "
            "demonstrated substantially lower accuracy/mAP in the wild than on PlantVillage, establishing that "
            "controlled-background models require domain adaptation and visual explainability audits."
        ),
    ]
    for title, desc in papers:
        add_subsec_heading(doc, title)
        add_body_p(doc, desc)

    add_sec_heading(doc, "2.3 Comparative Analysis of Existing Approaches")
    add_styled_table(
        doc,
        headers=["Author & Year", "Dataset", "Algorithm", "Performance", "Key Limitations"],
        rows=[
            ["Mohanty et al., 2016", "PlantVillage (54,306)", "AlexNet, GoogLeNet", "99.34% (GoogLeNet)", "Drops sharply on field backgrounds"],
            ["Sladojevic et al., 2016", "Custom web corpus", "CaffeNet CNN", "96.3% overall acc.", "Small unaugmented seed set; heavy FC head"],
            ["Brahimi et al., 2017", "PlantVillage Tomato (14,828)", "AlexNet, GoogLeNet + Saliency", "99.18% accuracy", "Single crop only; slow occlusion maps"],
            ["Ferentinos, 2018", "Extended PlantVillage (87,848)", "VGG, GoogLeNet, OverFeat", "99.53% (VGG)", "VGG too heavy (~138M params) for mobile"],
            ["Too et al., 2019", "PlantVillage (38 classes)", "DenseNet-121, ResNet, VGG16", "99.75% (DenseNet)", "No handcrafted feature baseline or leaf-group audit"],
            ["Singh et al., 2020", "PlantDoc (2,598 in-the-wild)", "Faster R-CNN, SSD", "Low in-wild baseline mAP", "Exposes lab-to-field domain gap"],
        ],
        col_widths=[1.2, 1.3, 1.3, 1.2, 1.5],
        font_size=8.5,
    )

    add_sec_heading(doc, "2.4 Research Gap")
    add_body_p(
        doc,
        "Three practical gaps emerge from prior work: (1) Many student and benchmark studies perform naive "
        "image-level random splits on PlantVillage without honoring physical leaf-group boundaries or "
        "deduplicating identical files, risking optimistic data leakage across Train and Test; (2) Most recent "
        "studies evaluate only deep CNNs without comparing domain-specific handcrafted computer vision features "
        "(color spaces, vegetation indices, GLCM, LBP) against neural networks on the exact same split; and "
        "(3) Direct binary Healthy-vs-Diseased classifiers are rarely compared against multi-class models "
        "rolled up to binary despite crop-species confounding in PlantVillage."
    )

    add_sec_heading(doc, "2.5 Proposed Solution")
    add_body_p(
        doc,
        "Our project directly addresses these gaps by: (a) removing 21 exact MD5 duplicates and enforcing "
        "StratifiedGroupKFold over 20,015 physical leaf groups using PlantVillage's leaf-map.json; (b) extracting "
        "114 interpretable color, vegetation, texture, and shape features to benchmark Logistic Regression, "
        "Random Forest, and a PyTorch Feature-MLP alongside a 1.21M-parameter Custom CNN and a 2.27M-parameter "
        "MobileNetV2 Transfer CNN; (c) comparing direct binary CNNs against 38-class-to-binary probability "
        "rollup; and (d) integrating Grad-CAM visual attention overlays inside a Streamlit diagnostic app."
    )

    # =========================================================================
    # CHAPTER 3: DATASET DESCRIPTION AND PREPROCESSING
    # =========================================================================
    add_chapter_heading(doc, "CHAPTER 3: DATASET DESCRIPTION AND PREPROCESSING", page_break=True)

    add_sec_heading(doc, "3.1 Dataset Description")
    add_body_p(
        doc,
        "The PlantVillage dataset is an open-access agricultural computer vision benchmark containing RGB "
        "photographs of individual healthy and diseased plant leaves captured under standardized lighting on "
        "neutral backgrounds. The dataset spans 14 crop species and 38 distinct crop–condition classes "
        "(12 healthy classes and 26 diseased classes covering fungal, bacterial, viral, oomycete, and mite infections)."
    )

    add_sec_heading(doc, "3.2 Dataset Source")
    add_body_p(
        doc,
        "Images and physical leaf-grouping metadata were retrieved directly from the official GitHub repository "
        "(https://github.com/spMohanty/PlantVillage-Dataset) using Git sparse checkout targeting raw/color "
        "and leaf_grouping/leaf-map.json."
    )

    add_sec_heading(doc, "3.3 Dataset Characteristics")
    add_styled_table(
        doc,
        headers=["Parameter", "Description"],
        rows=[
            ["Dataset Name", "PlantVillage Dataset (raw/color RGB subset)"],
            ["Source", "https://github.com/spMohanty/PlantVillage-Dataset"],
            ["Total Files Scanned / Usable", f"{dataset_summary['total_files_found']:,} scanned / {dataset_summary['usable_images']:,} unique usable (21 MD5 duplicates removed)"],
            ["Image Resolution", "256 x 256 pixels, 3-channel RGB JPEG (0 corrupt files)"],
            ["Total Features", "224 x 224 x 3 tensor (CNNs) and 114 engineered features (Feature-MLP & Classical ML)"],
            ["Target Variable", "Binary: Healthy (15,071) vs. Diseased (39,213); Multi-Class: 38 Crop___Condition labels"],
            ["Problem Type", "Supervised Image Classification (Binary & 38-Class Multi-Class)"],
        ],
        col_widths=[2.1, 4.4],
        font_size=9.5,
    )

    add_sec_heading(doc, "3.4 Feature Description")
    add_styled_table(
        doc,
        headers=["Feature Group", "Dims", "Data Type", "Description"],
        rows=[
            ["RGB Color Moments", "6", "Numerical (float32)", "Normalized mean & std of Red, Green, Blue channels"],
            ["HSV Color Features", "30", "Numerical (float32)", "Mean, std, and 8-bin normalized histograms for Hue, Saturation, Value"],
            ["CIELAB Color Features", "30", "Numerical (float32)", "Mean, std, and 8-bin histograms for L*, a* (green–red), b* (blue–yellow)"],
            ["Vegetation & Lesion Indices", "8", "Numerical (float32)", "ExG (2G-R-B) stats & green ratio, GRVI, yellow/brown, dark necrotic, a*>128 ratios"],
            ["GLCM Texture Descriptors", "12", "Numerical (float32)", "Contrast, dissimilarity, homogeneity, energy, correlation, ASM (d=1,3; 4 angles avg)"],
            ["Uniform LBP Texture", "18", "Numerical (float32)", "Rotation-invariant uniform Local Binary Patterns (P=16, R=2) normalized histogram"],
            ["Hu Moments & Edge Density", "10", "Numerical (float32)", "7 log-signed Hu shape moments + Sobel gradient mean/std + Canny edge ratio"],
            ["Target Labels", "2", "Categorical (int64)", "binary_id (0=healthy, 1=diseased) and class_id (0..37)"],
        ],
        col_widths=[1.7, 0.5, 1.3, 3.0],
        font_size=9,
    )

    add_sec_heading(doc, "3.5 Exploratory Data Analysis (EDA)")
    add_body_p(
        doc,
        "Exploratory analysis of the 54,284 unique images revealed three critical properties: (1) Binary class "
        "imbalance of 2.60:1 (39,213 diseased [72.2%] vs. 15,071 healthy [27.8%]); (2) Multi-class imbalance of "
        "36.2:1 between Orange___Haunglongbing_(Citrus_greening) (5,507 images) and Potato___healthy (152 images); "
        "and (3) Species–health confounding across 5 crops (Blueberry [1,502], Raspberry [371], and Soybean [5,090] "
        "contain only healthy leaves, whereas Orange [5,507] and Squash [1,835] contain only diseased leaves)."
    )
    add_figure(doc, figures_dir / "eda_binary_and_crop_distribution.png", "Figure 3.1: Binary Healthy vs. Diseased balance and crop-wise composition.", width_in=6.0)
    add_figure(doc, figures_dir / "eda_38_class_distribution.png", "Figure 3.2: Sample distribution across all 38 PlantVillage classes.", width_in=5.5)
    add_figure(doc, figures_dir / "eda_sample_leaves.png", "Figure 3.3: Representative healthy and diseased leaf images from PlantVillage.", width_in=5.8)

    add_sec_heading(doc, "3.6 Data Cleaning")
    add_body_p(
        doc,
        "Every file in raw/color was verified via Pillow header decoding (0 corrupt files detected) and hashed "
        "with MD5. Exactly 21 byte-identical duplicate images were removed prior to splitting, preventing "
        "identical samples from appearing in both training and evaluation folds. Physical leaf identifiers "
        "(leaf_id) were resolved using leaf-map.json combined with official PlantVillage copy-suffix stripping."
    )

    add_sec_heading(doc, "3.7 Feature Selection and Feature Engineering")
    add_body_p(
        doc,
        "All images were resized to 128 x 128 pixels and processed via PlantLeafFeatureExtractor to compute "
        "114 zero-NaN numerical descriptors capturing color shifts, chlorophyll indices, surface roughness, and "
        "lesion edges. Random Forest Gini importance confirmed that domain-engineered indices—led by "
        "veg_yellow_brown_ratio (0.1246 binary importance), hsv_h_hist_1 (0.0939), and uniform LBP bins—carry "
        "the highest discriminative signal."
    )
    add_figure(doc, figures_dir / "feature_importance_binary.png", "Figure 3.4: Top 20 handcrafted image features ranked by Random Forest Gini importance (Binary).", width_in=5.6)

    add_sec_heading(doc, "3.8 Data Transformation and Normalization")
    add_body_p(
        doc,
        "For tabular models (Logistic Regression, Random Forest, Feature-MLP), StandardScaler (zero mean, unit "
        "variance) was fit strictly on the Training split and applied identically to Validation and Test splits "
        "to avoid distribution leakage. For CNN pipelines, images were resized to 224 x 224 and normalized using "
        "ImageNet channel statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Training images "
        "received conservative geometry/color augmentations (horizontal/vertical flips, +/-20 deg rotation, mild "
        "ColorJitter with hue=0.02 so chlorosis colors are not distorted)."
    )

    add_sec_heading(doc, "3.9 Dataset Splitting")
    sc = dataset_summary["split_counts"]
    sl = dataset_summary["split_leaf_group_counts"]
    total_img = dataset_summary["usable_images"]
    add_styled_table(
        doc,
        headers=["Dataset Split", "Percentage", "Number of Images", "Unique Physical Leaf Groups", "Classes Covered"],
        rows=[
            ["Training", f"{100.0 * sc['train'] / total_img:.2f}%", f"{sc['train']:,}", f"{sl['train']:,}", "38 / 38"],
            ["Validation", f"{100.0 * sc['val'] / total_img:.2f}%", f"{sc['val']:,}", f"{sl['val']:,}", "38 / 38"],
            ["Testing", f"{100.0 * sc['test'] / total_img:.2f}%", f"{sc['test']:,}", f"{sl['test']:,}", "38 / 38"],
            ["Total", "100.00%", f"{total_img:,}", f"{dataset_summary['unique_leaf_groups']:,}", "38 / 38"],
        ],
        col_widths=[1.2, 1.0, 1.3, 1.7, 1.3],
        font_size=9.5,
    )

    # =========================================================================
    # CHAPTER 4: PROPOSED METHODOLOGY
    # =========================================================================
    add_chapter_heading(doc, "CHAPTER 4: PROPOSED METHODOLOGY", page_break=True)

    add_sec_heading(doc, "4.1 Overall System Architecture")
    add_body_p(
        doc,
        "The system implements a dual-branch experimental architecture operating on identical leaf-grouped "
        "splits so that handcrafted feature models and end-to-end convolutional neural networks can be compared fairly."
    )
    add_figure(doc, arch_png, "Figure 4.1: End-to-end architecture combining handcrafted image features (Pipeline A) and deep CNNs (Pipeline B).", width_in=6.2)

    add_sec_heading(doc, "4.2 Problem Formulation")
    add_body_p(
        doc,
        "Let an input RGB leaf image be I in R^(H x W x 3). In Pipeline A, a deterministic extractor phi(I) -> x in R^114 "
        "maps the image into a standardized feature vector. In Pipeline B, raw normalized tensors X in R^(3 x 224 x 224) "
        "are processed end-to-end. For multi-class classification, a classifier f_theta outputs logits z in R^38, "
        "normalized via softmax P(y = c | I) = exp(z_c) / sum_j exp(z_j). Let H_ids subset {0,...,37} be the 12 "
        "healthy class indices; the rolled-up Healthy vs. Diseased probability is computed as P(healthy | I) = "
        "sum_{c in H_ids} P(y = c | I), and P(diseased | I) = 1 - P(healthy | I)."
    )

    add_sec_heading(doc, "4.3 Machine Learning Algorithms")

    add_subsec_heading(doc, "4.3.1 Feature-MLP Neural Network (with Logistic Regression & Random Forest Baselines)")
    add_body_p(doc, "A 4-layer Multi-Layer Perceptron trained on the 114 standardized handcrafted image features alongside linear (Logistic Regression) and tree-ensemble (Random Forest, 200 trees) baselines.", bold_prefix="Introduction: ")
    add_body_p(doc, "Standardized image features pass sequentially through three hidden blocks—Linear(114->256), Linear(256->128), and Linear(128->64)—each paired with BatchNorm1d, ReLU activation, and decreasing Dropout (0.30, 0.20, 0.15) before the output linear layer.", bold_prefix="Working Principle: ")
    add_body_p(doc, "For layer l: h^(l) = Dropout(ReLU(BatchNorm(W^(l) h^(l-1) + b^(l)))), trained by minimizing class-weighted Cross-Entropy Loss L = -sum_c w_c y_c log(p_c) where w_c proportional to (N / n_c)^0.5.", bold_prefix="Mathematical Formulation: ")
    add_body_p(doc, "Advantages: Fast CPU/GPU training, compact (~0.3 MB) checkpoint, captures non-linear combinations of color and texture features. Limitations: Relies on fixed manual features and lacks spatial translation-invariant convolutions.", bold_prefix="Advantages & Limitations: ")

    add_subsec_heading(doc, "4.3.2 Custom Convolutional Neural Network Built from Scratch (PlantDiseaseCNN)")
    add_body_p(doc, "A 4-stage deep convolutional architecture designed and initialized from scratch (~1.21M parameters) trained directly on 224 x 224 leaf images.", bold_prefix="Introduction: ")
    add_body_p(doc, "Four hierarchical stages double feature channels (3 -> 32 -> 64 -> 128 -> 256) using stacked 3x3 convolutions, BatchNorm2d, ReLU, spatial Dropout2d, and 2x2 MaxPooling, followed by Global Average Pooling (AdaptiveAvgPool2d) and a 2-layer dense classifier head.", bold_prefix="Working Principle: ")
    add_body_p(doc, "Each 3x3 conv computes F_k(i,j) = ReLU(BN(sum_{m,u,v} K_{k,m}(u,v) X_m(i+u, j+v))). Global Average Pooling compresses spatial map F in R^(256 x 14 x 14) to g_k = (1/196) sum_{i,j} F_k(i,j), drastically curbing dense-layer parameter bloat.", bold_prefix="Mathematical Formulation: ")
    add_body_p(doc, "Advantages: Proves genuine domain-specific representation learning from scratch with low parameter count (1.21M). Limitations: Requires more epochs (12–15) than pretrained networks to learn primitive edge and texture filters.", bold_prefix="Advantages & Limitations: ")

    add_subsec_heading(doc, "4.3.3 Transfer Learning CNN (MobileNetV2) with Grad-CAM")
    add_body_p(doc, "Pretrained MobileNetV2 (IMAGENET1K_V2 weights, 2.27M parameters) fine-tuned end-to-end for 38-class leaf diagnosis, paired with Gradient-weighted Class Activation Mapping (Grad-CAM).", bold_prefix="Introduction: ")
    add_body_p(doc, "MobileNetV2 uses inverted residual blocks with linear bottlenecks and depthwise-separable convolutions to extract rich multi-scale features at minimal FLOPs; its final classifier is replaced by Dropout(0.30) + Linear(1280, 38).", bold_prefix="Working Principle: ")
    add_body_p(doc, "Depthwise-separable convolution factors a standard K x K x C_in x C_out kernel into spatial depthwise filtering (K x K x C_in) plus pointwise 1 x 1 mixing (C_in x C_out). Grad-CAM computes channel importance alpha_k^c = (1/Z) sum_{i,j} d(y^c)/d(A_{ij}^k) over the final conv activations A^k and maps L_GradCAM^c = ReLU(sum_k alpha_k^c A^k).", bold_prefix="Mathematical Formulation: ")
    add_body_p(doc, "Advantages: Achieves highest accuracy (99.64% 38-class, 99.95% rolled-up binary) in just 8 epochs while remaining lightweight (~9.3 MB checkpoint) and visually interpretable via Grad-CAM. Limitations: Pretrained ImageNet bias and PlantVillage lab-background sensitivity.", bold_prefix="Advantages & Limitations: ")

    add_sec_heading(doc, "4.4 Proposed ML Pipeline")
    for step_txt in [
        "Stage 1 (Acquisition & Integrity): Sparse-clone raw/color + leaf-map.json, verify readable RGB JPEGs, and drop 21 exact MD5 duplicates.",
        "Stage 2 (Leaf-Grouped Split): Partition 54,284 unique images via StratifiedGroupKFold(n_splits=7) over 20,015 leaf_id groups into 5:1:1 Train:Val:Test.",
        "Stage 3 (Dual Representation): Extract 114 handcrafted features (128x128) for Pipeline A; build augmented 224x224 PyTorch DataLoaders for Pipeline B.",
        "Stage 4 (Training & Checkpointing): Train Logistic Regression, Random Forest, Feature-MLP, Custom CNN (Binary & 38-class), and MobileNetV2 Transfer CNN, saving best validation checkpoints.",
        "Stage 5 (Evaluation & Deployment): Evaluate all models on the untouched 7,825-image Test split, compute rolled-up binary metrics, generate Grad-CAM visualizations, and serve via Streamlit.",
    ]:
        add_bullet(doc, step_txt)

    add_sec_heading(doc, "4.5 Algorithm / Pseudocode")
    add_code_block(
        doc,
        """Algorithm: Leakage-Free Plant Disease Classification & Binary Rollup
Input: PlantVillage raw/color images D, leaf_map M, epochs E_custom=12, E_transfer=8
1. Audit all images in D; remove corrupt files and duplicate MD5 hashes -> D_clean
2. For each image i in D_clean: assign leaf_id = ResolveLeafID(filename_i, M)
3. Split D_clean via StratifiedGroupKFold(n_splits=7, groups=leaf_id) -> Train (5 folds), Val (1), Test (1)
4. Extract 114-dim handcrafted features X_feat (RGB/HSV/LAB, ExG/GRVI/lesions, GLCM, LBP, Hu, Edges)
5. Fit StandardScaler S on X_feat[Train] only; transform Train, Val, Test
6. Train LogReg, RandomForest, and FeatureMLP on scaled X_feat (Binary & 38-Class)
7. Train CustomCNN and MobileNetV2 on 224x224 images using AMP, AdamW, CosineAnnealingLR, sqrt class weights
8. Select best checkpoints on Val accuracy/loss; evaluate once on Test
9. For 38-class CNN probabilities P in R^38: compute P(healthy) = sum_{c in HealthyIDs} P_c
10. Generate Grad-CAM heatmaps on final conv layer and deploy models in Streamlit UI.""",
    )

    add_sec_heading(doc, "4.6 Hyperparameter Tuning")
    add_styled_table(
        doc,
        headers=["Model", "Optimizer / Solver", "Initial LR", "Scheduler / Regularization", "Epochs / Trees"],
        rows=[
            ["Logistic Regression", "L-BFGS (sklearn)", "—", "L2 C=2.0, class_weight='balanced'", "max_iter=1000"],
            ["Random Forest", "CART Gini Ensemble", "—", "min_samples_leaf=2, balanced_subsample", "200 trees"],
            ["Feature-MLP", "AdamW (wd=1e-4)", "1e-3", "CosineAnnealingLR, Dropout(0.3/0.2/0.15), ES=6", "Up to 40 (ES ~30-32)"],
            ["Custom CNN (Binary)", "AdamW (wd=1e-4)", "1e-3", "CosineAnnealingLR, Dropout2d + Dropout(0.35), AMP", "15 epochs"],
            ["Custom CNN (38-Class)", "AdamW (wd=1e-4)", "1e-3", "Label smoothing=0.05, grad clip=2.0, AMP", "12 epochs"],
            ["MobileNetV2 (38-Class)", "AdamW (wd=1e-4)", "5e-4", "Label smoothing=0.05, Dropout(0.30), AMP", "8 epochs"],
        ],
        col_widths=[1.4, 1.2, 0.7, 2.0, 1.2],
        font_size=8.5,
    )

    add_sec_heading(doc, "4.7 Model Training and Validation")
    add_body_p(
        doc,
        "Every neural model monitored validation loss and validation accuracy after each epoch on the 7,718-image "
        "validation fold (2,858 distinct leaves). Checkpoints were persisted only upon validation improvement, "
        "and early stopping guarded against overfitting. Final test evaluation ran strictly once per model after "
        "training concluded."
    )

    # =========================================================================
    # CHAPTER 5: EXPERIMENTAL SETUP AND IMPLEMENTATION
    # =========================================================================
    add_chapter_heading(doc, "CHAPTER 5: EXPERIMENTAL SETUP AND IMPLEMENTATION", page_break=True)

    add_sec_heading(doc, "5.1 Hardware Requirements")
    add_styled_table(
        doc,
        headers=["Component", "Specification"],
        rows=[
            ["Processor (CPU)", "AMD Ryzen 7 7435HS (16 logical processors)"],
            ["RAM", "25.4 GB System Memory"],
            ["Storage", "432.8 GB NVMe SSD"],
            ["GPU", "NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM, CUDA Driver 591.66 / CUDA 13.1)"],
        ],
        col_widths=[2.2, 4.3],
        font_size=9.5,
    )

    add_sec_heading(doc, "5.2 Software Requirements")
    add_styled_table(
        doc,
        headers=["Software / Tool", "Version / Details"],
        rows=[
            ["Operating System", "Windows 11 (Windows-11-10.0.26300-SP0)"],
            ["Programming Language", "Python 3.13.5 (isolated .venv)"],
            ["IDE", "Visual Studio Code"],
            ["Deep Learning Framework", "PyTorch 2.11.0+cu128 & torchvision 0.26.0+cu128 (AMP enabled)"],
            ["ML & Image Processing Libraries", "scikit-learn 1.6.1, scikit-image 0.25.0, OpenCV 5.0.0, Pillow 11.1.0"],
            ["Data, Visualization & UI", "NumPy 2.1.3, Pandas 2.2.3, Matplotlib 3.10.0, Seaborn 0.13.2, Streamlit 1.45.1, pytest 8.3.4"],
        ],
        col_widths=[2.2, 4.3],
        font_size=9.5,
    )

    add_sec_heading(doc, "5.3 Implementation Details")
    add_body_p(
        doc,
        "The codebase is modularized inside src/ with centralized configuration in configs/config.yaml and "
        "orchestration via scripts/run_pipeline.py. Multiprocess feature extraction (ProcessPoolExecutor, 8 workers) "
        "processed all 54,284 images into compressed float32 matrices in ~149 seconds. DataLoaders utilize "
        "torchvision.transforms.v2 with pinned GPU memory and AMP float16 mixed-precision training. Eleven unit "
        "and smoke tests in tests/ verify feature invariants, model forward shapes, Grad-CAM hooks, leaf-ID "
        "normalization, and inference."
    )

    add_sec_heading(doc, "5.4 Model Training")
    add_body_p(
        doc,
        "Feature-MLP converged cleanly with early stopping at epoch 30 (Binary, best val_loss=0.0285) and epoch 32 "
        "(38-Class, best val_loss=0.1382). Custom CNN (Binary) trained 15 epochs reaching 99.13% validation "
        "accuracy; Custom CNN (38-Class) progressed from 69.19% val accuracy at Epoch 1 to 97.19% at Epoch 12; "
        "and MobileNetV2 Transfer CNN jumped from 98.16% val accuracy at Epoch 1 to 99.60% at Epoch 8 (~1,300.5s total)."
    )

    add_sec_heading(doc, "5.5 Model Testing")
    add_body_p(
        doc,
        "Saved checkpoints were evaluated on the held-out 7,825-image Test split (2,859 physical leaves) using "
        "deterministic eval transforms (Resize + Normalize only, no random augmentation). Predicted probability "
        "arrays were persisted for unified metric computation, ROC plotting, and error analysis."
    )

    add_sec_heading(doc, "5.6 Implementation Visuals (Training Curves)")
    add_figure(doc, figures_dir / "history_feature_mlp_binary.png", "Figure 5.1: Feature-MLP training and validation curves (Binary Healthy vs. Diseased).", width_in=5.8)
    add_figure(doc, figures_dir / "history_custom_cnn_multiclass.png", "Figure 5.2: Custom CNN (from scratch) training and validation curves (38-Class).", width_in=5.8)
    add_figure(doc, figures_dir / "history_transfer_mobilenet_v2_multiclass.png", "Figure 5.3: MobileNetV2 Transfer Learning CNN training and validation curves (38-Class).", width_in=5.8)

    # =========================================================================
    # CHAPTER 6: RESULTS AND PERFORMANCE EVALUATION
    # =========================================================================
    add_chapter_heading(doc, "CHAPTER 6: RESULTS AND PERFORMANCE EVALUATION", page_break=True)

    add_sec_heading(doc, "6.1 Experimental Results")
    add_body_p(
        doc,
        "All six modeling configurations were evaluated on the exact same 7,825 unseen test images. Handcrafted "
        "features combined with a neural network (Feature-MLP) strongly outperformed linear Logistic Regression "
        "and tree-based Random Forest on both Binary and 38-Class tasks, while convolutional neural networks "
        "delivered near-ceiling accuracy across all 38 plant–disease classes."
    )

    add_sec_heading(doc, "6.2 Performance Evaluation Metrics")
    for m_txt in [
        "Accuracy = (TP + TN) / (TP + TN + FP + FN): Overall proportion of correct predictions.",
        "Macro Precision = (1/C) sum_c TP_c / (TP_c + FP_c): Unweighted average positive predictive value across classes.",
        "Macro Recall = (1/C) sum_c TP_c / (TP_c + FN_c): Unweighted average sensitivity across all classes (critical under class imbalance).",
        "Macro F1-Score = (1/C) sum_c (2 * Prec_c * Rec_c) / (Prec_c + Rec_c): Harmonic mean balancing rare and frequent classes equally.",
        "Confusion Matrix & ROC-AUC: Detailed error breakdown across Healthy/Diseased and 38 classes plus threshold-independent binary ranking AUC.",
    ]:
        add_bullet(doc, m_txt)

    add_sec_heading(doc, "6.3 Performance Comparison")
    bench_rows = []
    for _, r in bench_df.iterrows():
        auc_str = f"{r['roc_auc']:.5f}" if pd.notna(r["roc_auc"]) else "—"
        bench_rows.append(
            [
                r["model"],
                r["task"],
                f"{r['accuracy']*100:.2f}%",
                f"{r['macro_precision']*100:.2f}%",
                f"{r['macro_recall']*100:.2f}%",
                f"{r['macro_f1']*100:.2f}%",
                auc_str,
            ]
        )
    add_styled_table(
        doc,
        headers=["Algorithm / Model", "Task", "Accuracy", "Macro Prec.", "Macro Rec.", "Macro F1", "ROC-AUC"],
        rows=bench_rows,
        col_widths=[1.8, 0.65, 0.75, 0.8, 0.8, 0.75, 0.95],
        font_size=8.5,
    )

    add_sec_heading(doc, "6.4 Graphical Analysis")
    add_figure(doc, figures_dir / "model_benchmark_comparison.png", "Figure 6.1: Held-out test Accuracy and Macro-F1 benchmark across all evaluated models.", width_in=5.9)
    add_figure(doc, figures_dir / "binary_roc_comparison.png", "Figure 6.2: Binary Healthy vs. Diseased ROC curve comparison across feature and CNN models.", width_in=5.1)
    add_figure(doc, figures_dir / "binary_confusion_matrices.png", "Figure 6.3: Binary confusion matrices (left-to-right: LogReg, Random Forest, Feature-MLP, Direct Binary Custom CNN, Custom CNN Rollup, MobileNetV2 Rollup).", width_in=6.2)
    add_figure(doc, figures_dir / "multiclass_confusion_transfer_cnn.png", "Figure 6.4: Row-normalized 38-class confusion matrix for MobileNetV2 Transfer CNN.", width_in=5.8)
    add_figure(doc, figures_dir / "gradcam_showcase.png", "Figure 6.5: Grad-CAM visual attention overlays verifying lesion-focused predictions on unseen test leaves.", width_in=5.3)

    add_sec_heading(doc, "6.5 Discussion and Interpretation")
    add_body_p(
        doc,
        "1. Neural Networks on Handcrafted Features: Feature-MLP improved 38-class accuracy from 91.55% (Random Forest) "
        "and 92.61% (Logistic Regression) to 96.10% (+4.55% over RF), confirming that dense non-linear feature "
        "interactions across color histograms, vegetation ratios, and LBP/GLCM texture outperform orthogonal tree splits.\n"
        "2. Multi-Class Rollup vs. Direct Binary CNN: Custom CNN (38-Class Rollup) achieved 99.71% binary accuracy "
        "(23 errors), outperforming Direct Binary Custom CNN (99.49%, 40 errors) with the identical backbone. "
        "Explicitly modeling 38 crop–disease targets avoids lumping 26 visually diverse pathologies into one heterogeneous class.\n"
        "3. Best Overall Model & Error Analysis: MobileNetV2 Transfer CNN achieved 99.64% 38-class accuracy (only "
        f"{error_analysis['total_misclassified']} errors out of {error_analysis['total_test_samples']:,} test images) and "
        "99.95% rolled-up binary accuracy (only 4 errors: 2 false positives, 2 false negatives). The primary confusion "
        "occurred between visually overlapping fungal corn blights (Cercospora gray leaf spot vs. Northern leaf blight: 6 errors)."
    )

    add_sec_heading(doc, "6.6 Limitations and Challenges")
    add_body_p(
        doc,
        "Key observed challenges include: (a) PlantVillage's uniform neutral backgrounds create domain shift when "
        "exposed to cluttered field soil or canopy photos; (b) 5 crops in PlantVillage contain only healthy or only "
        "diseased examples, risking species-level shortcuts in binary classifiers; (c) ~24% of PlantVillage filenames "
        "lack entries in leaf-map.json and rely on filename-stem grouping; and (d) closed-world softmax outputs cannot "
        "reject non-leaf uploads without an explicit out-of-distribution threshold."
    )

    # =========================================================================
    # CHAPTER 7: CONCLUSION AND FUTURE SCOPE
    # =========================================================================
    add_chapter_heading(doc, "CHAPTER 7: CONCLUSION AND FUTURE SCOPE", page_break=True)

    add_sec_heading(doc, "7.1 Conclusion")
    add_body_p(
        doc,
        "This mini-project successfully delivered an end-to-end, reproducible plant leaf disease classification "
        "system combining domain-engineered image features with neural networks. Enforcing physical leaf-grouped "
        "splitting prevented data leakage across all 54,284 images. Empirical evaluation on 7,825 unseen leaves "
        "demonstrated strong progression across modeling paradigms—from 93.10% (Logistic Regression) and 98.94% "
        "(Feature-MLP) using 114 image features, up to 99.64% 38-class and 99.95% rolled-up binary accuracy using "
        "MobileNetV2—supported by Grad-CAM explainability, 11 passing unit tests, and a working Streamlit app."
    )

    add_sec_heading(doc, "7.2 Achievement of Objectives")
    for ach in [
        "Problem formulated cleanly as both Binary (Healthy vs. Diseased) and 38-class crop–disease classification.",
        "PlantVillage audited (0 corrupt, 21 MD5 duplicates removed) and split strictly by 20,015 leaf groups.",
        "114 interpretable color, vegetation, texture (GLCM, LBP), and shape features engineered and validated.",
        "Classical baselines, Feature-MLP, Custom CNN from scratch, and MobileNetV2 Transfer CNN trained and compared.",
        "Grad-CAM explanations, publication plots, full documentation, and an interactive Streamlit web app delivered.",
    ]:
        add_bullet(doc, ach)

    add_sec_heading(doc, "7.3 Limitations")
    add_body_p(
        doc,
        "The deployed models remain constrained by laboratory-style single-leaf PlantVillage backgrounds, lack of "
        "healthy/diseased symmetry across 5 of the 14 species, single-label assumptions (ignoring co-infections or "
        "disease severity grading), and absence of an out-of-distribution non-leaf rejection gate."
    )

    add_sec_heading(doc, "7.4 Future Scope")
    for fut in [
        "In-the-Wild Domain Adaptation: Fine-tune on field datasets such as PlantDoc and apply CutMix background randomization using PlantVillage segmented masks.",
        "Out-of-Distribution Rejection: Add an energy-score / confidence-threshold non-leaf detector before classification.",
        "Severity Estimation & Multi-Label Diagnosis: Quantify infected leaf area percentage and handle co-occurring diseases.",
        "Mobile Edge Quantization: Export MobileNetV2 to INT8 TFLite / ONNX for offline smartphone diagnosis in low-connectivity farms.",
    ]:
        add_bullet(doc, fut)

    # =========================================================================
    # REFERENCES (IEEE Style)
    # =========================================================================
    add_chapter_heading(doc, "REFERENCES", page_break=True)
    refs = [
        '[1] S. P. Mohanty, D. P. Hughes, and M. Salathé, "Using deep learning for image-based plant disease detection," Frontiers in Plant Science, vol. 7, art. 1419, Sep. 2016.',
        '[2] S. Sladojević, M. Arsenović, A. Anderla, D. Culibrk, and D. Stefanović, "Deep neural networks based recognition of plant diseases by leaf image classification," Computational Intelligence and Neuroscience, vol. 2016, art. 3289801, 2016.',
        '[3] M. Brahimi, K. Boukhalfa, and A. Moussaoui, "Deep learning for tomato diseases: Classification and symptoms visualization," Applied Artificial Intelligence, vol. 31, no. 4, pp. 299–315, 2017.',
        '[4] K. P. Ferentinos, "Deep learning models for plant disease detection and diagnosis," Computers and Electronics in Agriculture, vol. 145, pp. 311–318, Feb. 2018.',
        '[5] E. C. Too, L. Yujian, S. Njuki, and L. Yingchun, "A comparative study of fine-tuning deep learning models for plant disease identification," Computers and Electronics in Agriculture, vol. 161, pp. 272–279, Jun. 2019.',
        '[6] D. Singh, N. Jain, P. Jain, P. Kayal, S. Kumawat, and N. Batra, "PlantDoc: A dataset for visual plant disease detection," in Proc. 7th ACM IKDD CoDS and 25th COMAD (CoDS-COMAD), 2020, pp. 249–253.',
        '[7] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen, "MobileNetV2: Inverted residuals and linear bottlenecks," in Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR), 2018, pp. 4510–4520.',
        '[8] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual explanations from deep networks via gradient-based localization," in Proc. IEEE Int. Conf. Computer Vision (ICCV), 2017, pp. 618–626.',
        '[9] R. M. Haralick, K. Shanmugam, and I. Dinstein, "Textural features for image classification," IEEE Trans. Systems, Man, and Cybernetics, vol. SMC-3, no. 6, pp. 610–621, Nov. 1973.',
        '[10] T. Ojala, M. Pietikäinen, and T. Mäenpää, "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns," IEEE Trans. Pattern Analysis and Machine Intelligence, vol. 24, no. 7, pp. 971–987, Jul. 2002.',
        '[11] PlantVillage Dataset Repository, spMohanty/PlantVillage-Dataset, GitHub. Available: https://github.com/spMohanty/PlantVillage-Dataset. Accessed: Nov. 27, 2026.',
    ]
    for ref in refs:
        add_body_p(doc, ref, space_after=5)

    # =========================================================================
    # APPENDIX A: SOURCE CODE
    # =========================================================================
    add_chapter_heading(doc, "APPENDIX A: SOURCE CODE", page_break=True)
    add_body_p(
        doc,
        "Full modular source code resides in the project repository under src/, scripts/, app/, and tests/. "
        "Below are core excerpts from the handcrafted feature extractor (src/features/extractor.py) and "
        "custom CNN architecture (src/models/custom_cnn.py)."
    )
    add_subsec_heading(doc, "A.1 Core Handcrafted Feature Extraction Excerpt (src/features/extractor.py)")
    add_code_block(
        doc,
        (PROJECT_ROOT / "src" / "features" / "extractor.py").read_text(encoding="utf-8")[:2600] + "\n# ... (truncated; see src/features/extractor.py for full implementation)",
    )
    add_subsec_heading(doc, "A.2 Custom CNN Architecture (src/models/custom_cnn.py)")
    add_code_block(
        doc,
        (PROJECT_ROOT / "src" / "models" / "custom_cnn.py").read_text(encoding="utf-8"),
    )

    # =========================================================================
    # APPENDIX B: ADDITIONAL EXPERIMENTAL RESULTS
    # =========================================================================
    add_chapter_heading(doc, "APPENDIX B: ADDITIONAL EXPERIMENTAL RESULTS", page_break=True)
    add_subsec_heading(doc, "B.1 Per-Class Classification Report — MobileNetV2 (38 Classes on Test Split)")
    cls_rows = []
    for cls_name in report_38_df.index[:38]:
        r = report_38_df.loc[cls_name]
        cls_rows.append(
            [
                cls_name.replace("___", " : "),
                f"{r['precision']*100:.2f}%",
                f"{r['recall']*100:.2f}%",
                f"{r['f1-score']*100:.2f}%",
                f"{int(r['support'])}",
            ]
        )
    add_styled_table(
        doc,
        headers=["PlantVillage Class (Crop : Condition)", "Precision", "Recall", "F1-Score", "Support"],
        rows=cls_rows,
        col_widths=[3.3, 0.8, 0.8, 0.8, 0.8],
        font_size=7.5,
    )

    add_subsec_heading(doc, "B.2 Top Misclassified Class Pairs (MobileNetV2, 28 Errors / 7,825 Samples)")
    err_rows = [
        [e["true_class"].replace("___", " : "), e["pred_class"].replace("___", " : "), str(e["count"])]
        for e in error_analysis["top_confused_pairs"][:10]
    ]
    add_styled_table(
        doc,
        headers=["True Class", "Predicted Class", "Error Count"],
        rows=err_rows,
        col_widths=[2.75, 2.75, 1.0],
        font_size=8.5,
    )
    add_figure(doc, figures_dir / "feature_importance_multiclass.png", "Figure B.1: Top 20 handcrafted image features for 38-class Random Forest.", width_in=5.5)
    add_figure(doc, figures_dir / "multiclass_confusion_custom_cnn.png", "Figure B.2: Row-normalized 38-class confusion matrix for Custom CNN (from scratch).", width_in=5.6)

    # =========================================================================
    # APPENDIX C: PROJECT OUTPUTS
    # =========================================================================
    add_chapter_heading(doc, "APPENDIX C: PROJECT OUTPUTS", page_break=True)
    add_body_p(
        doc,
        "All models were verified via automated pytest unit tests (11/11 passing), Streamlit headless AppTest "
        "verification (3 interactive tabs loaded with zero exceptions), and single-image inference tests via "
        "PlantDiseasePredictor."
    )

    from src.inference.predictor import PlantDiseasePredictor

    predictor = PlantDiseasePredictor()
    test_df = splits_df[splits_df["split"] == "test"].reset_index(drop=True)
    sample_rows = []
    # Pick 6 diverse crops from test split
    for idx in [0, 350, 1200, 2600, 4500, 6500]:
        if idx < len(test_df):
            row = test_df.iloc[idx]
            res = predictor.predict(PROJECT_ROOT / row["rel_path"], model_choice="Transfer CNN (mobilenet_v2)")
            sample_rows.append(
                [
                    row["class_name"].replace("___", " : "),
                    res["top_class"].replace("___", " : "),
                    res["binary_verdict"],
                    f"{res['top_confidence']*100:.2f}%",
                    f"{res['key_image_features']['Excess Green Healthy Ratio']:.3f}",
                    f"{res['key_image_features']['Yellow/Brown Chlorosis Ratio']:.3f}",
                ]
            )
    add_styled_table(
        doc,
        headers=["Ground Truth", "Predicted Class", "Verdict", "Confidence", "ExG Green Ratio", "Chlorosis Ratio"],
        rows=sample_rows,
        col_widths=[1.65, 1.65, 0.7, 0.8, 0.85, 0.85],
        font_size=8,
    )
    add_body_p(
        doc,
        "Interactive Application Launch Command: .\\.venv\\Scripts\\python.exe -m streamlit run app/app.py\n"
        "Automated Test Suite Result: 11 passed in tests/ (feature extraction, CNN forward passes, Grad-CAM, splits, and predictor smoke test)."
    )

    out_docx = docx_dir / "Plant_Disease_Classification_Project_Report.docx"
    doc.save(str(out_docx))
    print(f"Generated Word project report successfully: {out_docx}")
    return out_docx


if __name__ == "__main__":
    build_report()

# app_streamlit.py
import streamlit as st
from pathlib import Path
import tempfile
import pandas as pd
import re
import argparse
from typing import List, Dict
from io import BytesIO

# ---- extractor functions (adapted from extract_kv_comments_fixed.py) ----

try:
    import PyPDF2
except Exception as e:
    PyPDF2 = None
    # We will raise if user tries to process without PyPDF2
    # but allow app to start so user sees instructions.

def read_pdf_text(path: Path) -> str:
    """Read text-layer from PDF using PyPDF2."""
    if PyPDF2 is None:
        raise RuntimeError("PyPDF2 not installed. Install with: pip install pypdf2")
    text_pages = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            ptxt = p.extract_text() or ""
            text_pages.append(ptxt)
    return "\n".join(text_pages)

def paragraphs_from_text(text: str) -> List[str]:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    parts = re.split(r'\n\s*\n+', text)
    paras = []
    for p in parts:
        p_clean = re.sub(r'\s+', ' ', p).strip()
        if p_clean:
            paras.append(p_clean)
    return paras

def extract_from_paragraph(par: str) -> List[Dict]:
    entries = []
    kv_pattern = re.compile(r'^\s*([A-Za-z0-9 \-_\./\(\)&]+?)\s*[:\-–—]\s*(.+)$')
    kv_matches = list(kv_pattern.finditer(par))
    if kv_matches:
        for m in kv_matches:
            key = m.group(1).strip()
            value = m.group(2).strip()
            entries.append({"Key": key, "Value": value, "Comments": m.group(0).strip()})
        return entries

    patterns = [
        (re.compile(r'([A-Z][a-z]+(?: [A-Z][a-z]+){0,3}) was born on ([A-Za-z0-9 ,\-]+)', flags=re.I), ("Name", "Date_of_Birth_verbose")),
        (re.compile(r'\bDOB\b[:\s]*([0-9A-Za-z ,\-]+)', flags=re.I), ("Date_of_Birth",)),
        (re.compile(r'\bDate of Birth\b[:\s]*([0-9A-Za-z ,\-]+)', flags=re.I), ("Date_of_Birth",)),
        (re.compile(r'\bmaking (?:him|her|them) (\d{1,3}) years', flags=re.I), ("Age",)),
        (re.compile(r'joined (?:his|her)? ?(?:first )? company as a ([A-Za-z0-9 \-&,]+?) with an annual salary of ([\d,]+) ?INR', flags=re.I), ("Employment_Role", "Employment_Salary")),
        (re.compile(r'current role at ([A-Za-z0-9 \-&]+) beginning on ([A-Za-z0-9 ,]+), where he serves as a ([A-Za-z0-9 \-&]+) earning ([\d,]+) ?INR', flags=re.I),
         ("Employment_Company", "Employment_Start", "Employment_Role", "Employment_Salary")),
        (re.compile(r'worked at ([A-Za-z0-9 \-&]+) from ([A-Za-z0-9 ,\d]+) to ([A-Za-z0-9 ,\d]+), starting as a ([A-Za-z0-9 \-&]+)(?: and earning a promotion in (\d{4}))?', flags=re.I),
         ("Employment_Company", "Employment_From", "Employment_To", "Employment_Role", "Employment_PromotionYear")),
        (re.compile(r'completed his 12th standard in (\d{4}), achieving an? ([0-9\.%]+)', flags=re.I), ("HighSchool_Year", "HighSchool_Score")),
        (re.compile(r'B\.?Tech in ([A-Za-z0-9 \-&]+) at ([A-Za-z0-9 \-&]+), graduating .* in (\d{4}) with a CGPA of ([\d\.]+)', flags=re.I),
         ("BTech_Degree", "BTech_Institution", "BTech_Graduation_Year", "BTech_CGPA")),
        (re.compile(r'M\.?Tech in ([A-Za-z0-9 \-&]+) in (\d{4}), .* CGPA of ([\d\.]+)', flags=re.I), ("MTech_Degree", "MTech_Graduation_Year", "MTech_CGPA")),
        (re.compile(r'(\bO\+|\bA\+|\bB\+|AB\+)\b.*blood', flags=re.I), ("Blood_Group",)),
    ]

    matched_any = False
    for pat, keys in patterns:
        for m in pat.finditer(par):
            matched_any = True
            groups = m.groups()
            for i, key_name in enumerate(keys):
                val = groups[i] if i < len(groups) and groups[i] is not None else ""
                entries.append({"Key": key_name, "Value": str(val).strip(), "Comments": m.group(0).strip()})
    if matched_any:
        return entries

    skill_pat = re.compile(r'([A-Za-z0-9 &\+]+?) (?:expertise|proficiency|proficient|scores?) (?:at )?([0-9]{1,2}) out of ([0-9]{1,2})', flags=re.I)
    for m in skill_pat.finditer(par):
        skill = m.group(1).strip()
        rating = f"{m.group(2)}/{m.group(3)}"
        entries.append({"Key": f"Skill_{skill}", "Value": rating, "Comments": m.group(0).strip()})
    if entries:
        return entries

    if re.search(r'\b(joined|worked at|current role at|currently at)\b', par, flags=re.I):
        entries.append({"Key": "Employment", "Value": par, "Comments": par})
        return entries

    entries.append({"Key": "Text_Line", "Value": par, "Comments": ""})
    return entries

def dedupe_entries(entries: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for e in entries:
        pair = (e.get("Key", ""), e.get("Value", ""))
        if pair in seen:
            continue
        seen.add(pair)
        out.append(e)
    return out

def build_df_and_excel(entries: List[Dict]) -> (pd.DataFrame, bytes):
    df = pd.DataFrame(entries, columns=["Key", "Value", "Comments"])
    if "Comments" not in df.columns:
        df["Comments"] = ""
    # Prepare in-memory excel bytes
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    return df, output.getvalue()

# ---- Streamlit UI ----

st.set_page_config(page_title="AI Document Structurer", layout="wide", page_icon="📄")

st.markdown("<h1 style='text-align:center'>AI Document Structurer</h1>", unsafe_allow_html=True)
st.write("Upload a PDF or use the sample file. The app extracts Key / Value / Comments rows and shows a downloadable Excel.")

col1, col2 = st.columns([2,1])

with col1:
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], help="Drop your Data Input PDF here")
    # use_sample = st.button("Use sample PDF on disk")
    st.markdown("**Note:** If you don't upload a file, you can use the sample file that is already on the server.")
    # show link to the sample file (local path). Developer asked to include the path as URL.
    sample_path = Path("/mnt/data/Data Input.pdf")
    if sample_path.exists():
        # file:// URL will allow users on the same machine to open it, otherwise it's a hint
        st.markdown(f"**Sample file path:** `{sample_path}` — [Open local file](file://{sample_path})")

with col2:
    st.sidebar.header("Options")
    show_original = st.sidebar.checkbox("Include Original_Text row in output", value=True)
    max_preview = st.sidebar.slider("Preview rows (top N)", min_value=5, max_value=200, value=25)
    pretty_style = st.sidebar.checkbox("Compact table display", value=True)

process_clicked = st.button("Process PDF")

# Choose file to use
input_path = None
temp_file_path = None
if uploaded is not None:
    # Save uploaded bytes to a temp file so PyPDF2 can read
    tfd = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tfd.write(uploaded.read())
    tfd.flush()
    temp_file_path = Path(tfd.name)
    input_path = temp_file_path
    st.success(f"Uploaded file saved to temporary path: {input_path}")
# elif use_sample:
#     if sample_path.exists():
#         input_path = sample_path
#         st.success(f"Using sample file: {input_path}")
#     else:
#         st.error("Sample file not found on server.")
#         input_path = None

# Process if asked
if process_clicked:
    if input_path is None:
        st.error("Please upload a PDF or click 'Use sample PDF on disk'")
    else:
        try:
            with st.spinner("Extracting text from PDF..."):
                raw_text = read_pdf_text(Path(input_path))
            if not raw_text or len(raw_text.strip()) < 10:
                st.error("Extracted text is empty or unreadable. If your PDF is scanned, enable OCR fallback (not included in this demo).")
            else:
                paras = paragraphs_from_text(raw_text)
                st.info(f"Detected {len(paras)} paragraphs after normalization.")
                all_entries = []
                progress_bar = st.progress(0)
                for idx, p in enumerate(paras):
                    extracted = extract_from_paragraph(p)
                    all_entries.extend(extracted)
                    # update progress
                    if len(paras) > 0:
                        progress_bar.progress(min(100, int((idx+1)/len(paras)*100)))
                # append original text optionally
                if show_original:
                    full_text_normalized = " ".join(raw_text.split())
                    all_entries.append({"Key":"Original_Text","Value":full_text_normalized,"Comments":"Full verbatim PDF content."})
                final = dedupe_entries(all_entries)
                df, excel_bytes = build_df_and_excel(final)
                st.success(f"Extraction complete — {len(df)} rows produced.")
                # Display table
                st.markdown("### Preview (Key | Value | Comments)")
                if pretty_style:
                    st.dataframe(df.head(max_preview), use_container_width=True)
                else:
                    st.table(df.head(max_preview))
                # Download button
                btn_label = "Download Excel"
                st.download_button(btn_label, excel_bytes, file_name="Output_key_value_comments.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                # Optionally show JSON of a few rows for debugging
                with st.expander("Show first 5 rows as JSON"):
                    st.json(df.head(5).to_dict(orient="records"))
        except Exception as e:
            st.error(f"Processing failed: {e}")

# Cleanup temp file when app finishes session (optional)
if temp_file_path is not None:
    try:
        Path(temp_file_path).unlink(missing_ok=True)
    except Exception:
        pass

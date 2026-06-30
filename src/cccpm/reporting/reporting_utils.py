import base64
import os
import re
from pathlib import Path

import pandas as pd


def format_results_table(df, precision=2):
    """
    Format a MultiIndex DataFrame:
    - Merge 'mean' and 'std' into 'summary'
    - Format p-values: add * / ** and highlight in bold using CSS
    - Return a styled Pandas DataFrame with APA style
    """
    formatted = {}
    metrics = df.columns.get_level_values(0).unique()

    for metric in metrics:
        mean = df[(metric, "mean")]
        std = df[(metric, "std")]
        p = df[(metric, "p")]

        # Format mean [std]
        summary_col = mean.round(precision).astype(str) + " [" + std.round(precision).astype(str) + "]"

        # Annotate p-values with asterisks (we'll apply bold via styling)
        def p_string(val):
            if pd.isna(val):
                return ""
            elif val < 0.001:
                return "<0.001**"
            elif val < 0.01:
                return f"{val:.3f}**"
            elif val < 0.05:
                return f"{val:.3f}*"
            else:
                return f"{val:.3f}"

        formatted[(metric, "mean [sd]")] = summary_col
        formatted[(metric, "p")] = p.apply(p_string)

    combined = pd.DataFrame(formatted, index=df.index)
    combined.columns = pd.MultiIndex.from_tuples(combined.columns)

    # Column sort: summary → p
    combined = combined.loc[:, sorted(combined.columns, key=lambda x: (x[0], ["mean [sd]", "p"].index(x[1])))]

    # Build Styler
    styler = combined.style.set_properties(
        **{
            'font-size': '10px',
            'padding': '2px 4px',
            'text-align': 'center'
        }
    ).set_table_styles([
        {'selector': 'th',
         'props': [('font-size', '11px'),
                   ('padding', '2px 4px'),
                   ('text-align', 'center'),
                   ('background-color', '#f9f9f9')]},
        {'selector': '.row_heading',
         'props': [('font-size', '10px'),
                   ('padding', '2px 4px')]},
        {'selector': '.index_name',
         'props': [('font-size', '10px'),
                   ('padding', '2px 4px')]}
    ])

    # Apply bold to significant p-values via CSS
    def bold_sig(val):
        if isinstance(val, str) and val.endswith("**") or val.endswith("*"):
            return 'font-weight: bold'
        return ''

    # Apply only to p-value columns
    for col in combined.columns:
        if col[1] == "p":
            styler = styler.map(bold_sig, subset=[col])
    # Add thick horizontal lines between top-level index groups
    def thick_divider_rows(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        previous_group = None
        for i, idx in enumerate(df.index):
            current_group = idx[0]  # assumes 'model' is the first index level
            if previous_group is not None and current_group != previous_group:
                styles.iloc[i] = 'border-top: 1px solid black'
            previous_group = current_group
        return styles

    styler = styler.apply(thick_divider_rows, axis=None)
    return styler


def extract_log_block(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Find all indices of separator lines (e.g. "=======")
    sep_indices = [i for i, line in enumerate(lines) if line.strip().startswith("=")]

    if len(sep_indices) >= 2:
        # Take everything between the first two separator lines
        start = sep_indices[0] + 1
        end = sep_indices[1]
        content = lines[start:end]
    else:
        content = []  # or raise an error, depending on your expectations

    return "".join(content).strip()


# Function to read CSV file from the given folder path
def load_data_from_folder(folder_path, filename):
    csv_path = os.path.join(folder_path, filename)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        raise RuntimeError(f"No CSV file found at path: {csv_path}")


def embed_svg(path: str) -> str:
    """Read an SVG file and return its content as an inline string."""
    p = Path(path)
    if not p.exists():
        return f'<p class="missing-asset">Figure not found: {p.name}</p>'
    return p.read_text(encoding="utf-8")


def embed_image_base64(path: str, mime: str = "image/png") -> str:
    """Return an <img> tag with the image embedded as a base64 data URI."""
    p = Path(path)
    if not p.exists():
        return f'<p class="missing-asset">Image not found: {p.name}</p>'
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f'<img src="data:{mime};base64,{b64}" style="max-width:100%;" />'


def parse_config_block(log_path: str) -> list[tuple[str, str]]:
    """
    Parse the configuration block from cpm_log.txt.

    Returns a list of (key, value) pairs in log order.
    """
    raw = extract_log_block(log_path)
    pairs = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([^:]+):\s*(.*)$", line)
        if m:
            pairs.append((m.group(1).strip(), m.group(2).strip()))
        else:
            pairs.append((line, ""))
    return pairs


def styler_to_html(styler) -> str:
    """Render a pandas Styler to an HTML string."""
    return styler.to_html(escape=False)


def load_results_from_folder(folder_path, filename):
    csv_path = os.path.join(folder_path, filename)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, header=[0, 1], index_col=[0, 1, 2])
    else:
        raise RuntimeError(f"No CSV file found at path: {csv_path}")

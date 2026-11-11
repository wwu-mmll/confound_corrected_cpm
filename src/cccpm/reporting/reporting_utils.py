import base64
import os
import re
from pathlib import Path

import pandas as pd


def average_over_repeats(df, value_cols):
    """Collapse a per-fold frame to one row per (subject, model, network), averaging value_cols.

    A RepeatedKFold outer CV tests each subject once per repeat, so this is a
    no-op for plain CV but collapses repeat-duplicates otherwise. Returned
    unchanged if ``df`` lacks the grouping keys.
    """
    keys = ['sample_index', 'model', 'network']
    if df is None or df.empty or not set(keys).issubset(df.columns):
        return df
    agg = {c: 'mean' for c in value_cols if c in df.columns}
    if 'y_true' in df.columns:
        agg['y_true'] = 'first'
    return df.groupby(keys, as_index=False).agg(agg)


def format_results_table(df, precision=2):
    """Format a MultiIndex results frame into an APA-style table: merges mean/std into "mean [sd]", bolds significant p-values."""
    formatted = {}
    metrics = df.columns.get_level_values(0).unique()

    for metric in metrics:
        mean = df[(metric, "mean")]
        std = df[(metric, "std")]
        p = df[(metric, "p")]

        # NaN means "not applicable" (non-incrementable metric, or an undefined
        # model variant), not "missing" -- an em dash says that, "nan" does not.
        def summary_string(row):
            if pd.isna(row["mean"]):
                return "\u2014"
            if pd.isna(row["std"]):
                return f"{row['mean']:.{precision}f}"
            return f"{row['mean']:.{precision}f} [{row['std']:.{precision}f}]"

        summary_col = pd.DataFrame({"mean": mean, "std": std}).apply(
            summary_string, axis=1)

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

    combined = combined.loc[:, sorted(combined.columns, key=lambda x: (x[0], ["mean [sd]", "p"].index(x[1])))]

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

    def bold_sig(val):
        if isinstance(val, str) and val.endswith("**") or val.endswith("*"):
            return 'font-weight: bold'
        return ''

    for col in combined.columns:
        if col[1] == "p":
            styler = styler.map(bold_sig, subset=[col])

    def thick_divider_rows(df):
        """Border above each row where the top-level ('model') index changes."""
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
    """Extract the text between the first two "=" separator lines of a log file."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sep_indices = [i for i, line in enumerate(lines) if line.strip().startswith("=")]

    if len(sep_indices) >= 2:
        start = sep_indices[0] + 1
        end = sep_indices[1]
        content = lines[start:end]
    else:
        content = []

    return "".join(content).strip()


def load_data_from_folder(folder_path, filename):
    """Read a CSV file from a results folder; raises RuntimeError if missing."""
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
    """Parse cpm_log.txt's configuration block into (key, value) pairs, in log order."""
    raw = extract_log_block(log_path)
    pairs = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        # Indented continuation of a multi-line value (e.g. an estimator repr),
        # not a new key -- else it produces rows like "threshold=[0.05])]," with
        # an empty value.
        is_continuation = line[:1].isspace() and pairs
        line = line.strip()
        m = re.match(r"^([^:]+):\s*(.*)$", line)
        if is_continuation:
            key, value = pairs[-1]
            pairs[-1] = (key, f"{value} {line}".strip())
        elif m:
            pairs.append((m.group(1).strip(), m.group(2).strip()))
        else:
            pairs.append((line, ""))
    return pairs


def styler_to_html(styler) -> str:
    """Render a pandas Styler to an HTML string."""
    return styler.to_html(escape=False)


def load_results_from_folder(folder_path, filename):
    """Read a two-level-header, three-level-index summary CSV; raises RuntimeError if missing."""
    csv_path = os.path.join(folder_path, filename)
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, header=[0, 1], index_col=[0, 1, 2])
    else:
        raise RuntimeError(f"No CSV file found at path: {csv_path}")

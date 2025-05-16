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

def table_to_latex(df, caption, label, out_path):
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:{label}}}",
        df.to_latex(index=False, escape=False),
        "\\end{table}"
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
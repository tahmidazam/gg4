"""Shared utilities for matplotlib figure export to PGF.

Usage pattern (DRY: same make_figure call for both display and export):

    # display cell
    configure_screen()
    fig = make_figure(data, figsize=(8, 4))
    plt.show()

    # pgf export cell
    FIG_WIDTH_FRAC = 1.0
    FIG_HEIGHT_FRAC = 0.3
    configure_pgf()
    fig = make_figure(data, figsize=figure_inches(FIG_WIDTH_FRAC, FIG_HEIGHT_FRAC))
    save_pgf(fig, FIGURES_PATH / "name.pgf")
    plt.close(fig)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

# A4 page content dimensions with 1-inch margins (inches)
A4_W_IN: float = 6.26
A4_H_IN: float = 9.69

_SCREEN_PARAMS: dict = {
    "text.usetex": False,
    "font.family": "sans-serif",
}

_PGF_PARAMS: dict = {
    "pgf.texsystem": "pdflatex",
    "font.family": "sans-serif",
    "text.usetex": True,
    "pgf.preamble": (
        r"\usepackage{amsmath}"
        r"\usepackage{amssymb}"
        r"\renewcommand{\familydefault}{\sfdefault}"
        r"\usepackage{sfmath}"
    ),
}


def configure_screen() -> None:
    """Set rcParams for on-screen notebook display."""
    mpl.rcParams.update(_SCREEN_PARAMS)


def configure_pgf() -> None:
    """Set rcParams for PGF/LaTeX export with sans-serif fonts throughout."""
    mpl.rcParams.update(_PGF_PARAMS)


def figure_inches(width_frac: float, height_frac: float) -> tuple[float, float]:
    """Return (width_in, height_in) as fractions of A4 content dimensions."""
    return width_frac * A4_W_IN, height_frac * A4_H_IN


def apply_figure_style(fig: mpl.figure.Figure) -> None:
    """Apply journal-style formatting to every axes in a figure.

    Centralises styling that would otherwise be repeated per-axis in every
    make_*_figure function: left-aligned small titles, small axis labels,
    major/minor gridlines (skipped for image axes), and small legend text.
    Called automatically by save_pgf; call it manually in display cells too
    if consistent styling is wanted on-screen.
    """
    for ax in fig.axes:
        title_text = ax.get_title()
        ax.set_title(title_text, loc="left", fontsize="small")
        ax.set_title("", loc="center")
        ax.xaxis.label.set_size("small")
        ax.yaxis.label.set_size("small")
        ax.tick_params(axis="both", labelsize="small")
        if not ax.images:
            ax.minorticks_on()
            ax.grid(True, which="major", linestyle="-", alpha=0.5)
            ax.grid(True, which="minor", linestyle=":", alpha=0.3)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize("x-small")
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontsize("x-small")


def save_pgf(fig: mpl.figure.Figure, path: Path | str, dpi: int = 150) -> None:
    """Save a figure to PGF, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    apply_figure_style(fig)
    fig.savefig(path, backend="pgf", dpi=dpi)
    print(f"Saved to {path.resolve()}")


def notebook_github_url(source_file: str | Path) -> str:
    """Return the GitHub blob URL for source_file.

    Pass ``__file__`` from a companion ``.py`` script; ``.py`` is swapped for
    ``.ipynb`` automatically.  The remote URL and repository root are resolved
    via ``git``.
    """
    import subprocess
    path = Path(source_file).resolve()
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=path.parent
        ).decode().strip()
    ).resolve()
    rel = path.relative_to(root)
    if rel.suffix == ".py":
        rel = rel.with_suffix(".ipynb")
    remote = (
        subprocess.check_output(
            ["git", "remote", "get-url", "origin"], cwd=root
        )
        .decode()
        .strip()
    )
    base = remote.removesuffix(".git").replace(
        "git@github.com:", "https://github.com/"
    )
    return f"{base}/blob/main/{rel}"


def save_table(path: Path | str, tabular: str, source_url: str) -> None:
    r"""Write a LaTeX tabular to a snippet file for \input into a table float.

    Appends a source hyperlink as a final table note so the origin is
    traceable from the PDF without cluttering the caption.
    A trailing % suppresses the end-of-file newline artefact.
    """
    source = "\n" + r"\par\smallskip\raggedleft{\tiny\href{" + source_url + r"}{\underline{Source}}}"
    content = tabular.rstrip() + source + "%"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")
    print(f"Saved table to {Path(path).resolve()}")


def save_caption(path: Path | str, caption: str, source_url: str) -> None:
    r"""Write a caption to a LaTeX snippet file, appending a source hyperlink.

    The file is designed to be \input-ed inside \caption{} in main.tex.
    A trailing % suppresses the end-of-file newline that would otherwise
    produce a spurious space inside the rendered caption.
    """
    source = r"\href{" + source_url + r"}{\underline{Source}}"
    content = caption.rstrip() + " " + source + "%"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")
    print(f"Saved caption to {Path(path).resolve()}")

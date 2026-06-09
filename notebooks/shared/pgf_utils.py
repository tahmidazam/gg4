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


def save_pgf(fig: mpl.figure.Figure, path: Path | str) -> None:
    """Save a figure to PGF, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, backend="pgf")
    print(f"Saved to {path.resolve()}")


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

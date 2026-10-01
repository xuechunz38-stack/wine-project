"""Console formatting shared by the command-line scripts."""

from __future__ import annotations


def print_section(title: str, width: int = 70) -> None:
    """Print a section header such as '==== 2. INSPECTING THE DATA ===='."""
    rule = "=" * width
    print(f"\n{rule}\n{title}\n{rule}")

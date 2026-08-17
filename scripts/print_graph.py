"""Regenerate the Mermaid diagram in README.md from the compiled graph.

The diagram is generated, not drawn, so it can't drift from the code. Run
`make graph` after changing the graph and commit the result.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eightysix.graph import build_graph

README = Path(__file__).resolve().parent.parent / "README.md"
START, END = "<!-- graph:start -->", "<!-- graph:end -->"

if __name__ == "__main__":
    mermaid = build_graph().get_graph().draw_mermaid().strip()
    block = f"{START}\n\n```mermaid\n{mermaid}\n```\n\n{END}"

    text = README.read_text()
    if START in text and END in text:
        text = re.sub(
            re.escape(START) + r".*?" + re.escape(END), block, text, flags=re.DOTALL
        )
        README.write_text(text)
        print(f"updated {README.name}")
    else:
        print(mermaid)
        print(f"\n(no {START} / {END} markers in README.md — printed instead)")

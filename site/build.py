"""Build the results page: data.json (from build_data.py) inlined into template.html.

Writes site/i-spy-agent.html (local preview) and docs/index.html (served by GitHub Pages).
"""
from pathlib import Path

here = Path(__file__).resolve().parent
data = (here / "data.json").read_text().replace("</", "<\\/")
page = (here / "template.html").read_text().replace("__DATA__", data)
(here / "i-spy-agent.html").write_text(page)
(here.parent / "docs").mkdir(exist_ok=True)
(here.parent / "docs" / "index.html").write_text(page)

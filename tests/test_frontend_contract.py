import re
from html.parser import HTMLParser
from pathlib import Path


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


class FrontendHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.script_srcs: list[str] = []
        self.link_hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        if element_id := attr_map.get("id"):
            self.ids.add(element_id)
        if tag == "script" and attr_map.get("src"):
            self.script_srcs.append(attr_map["src"] or "")
        if tag == "link" and attr_map.get("href"):
            self.link_hrefs.append(attr_map["href"] or "")


def _parse_index() -> FrontendHtmlParser:
    parser = FrontendHtmlParser()
    parser.feed((FRONTEND_DIR / "index.html").read_text(encoding="utf-8"))
    return parser


def test_frontend_core_controls_are_self_contained() -> None:
    parser = _parse_index()
    external_assets = [
        asset
        for asset in [*parser.script_srcs, *parser.link_hrefs]
        if asset.startswith(("http://", "https://", "//"))
    ]

    assert external_assets == []


def test_javascript_dom_selectors_exist_in_html() -> None:
    parser = _parse_index()
    script = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")
    referenced_ids = set(re.findall(r"getElementById\(['\"]([^'\"]+)['\"]\)", script))

    assert referenced_ids - parser.ids == set()


def test_chat_rendering_does_not_require_marked_or_dompurify_globals() -> None:
    script = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")

    assert "new marked.Renderer" not in script
    assert "DOMPurify" not in script

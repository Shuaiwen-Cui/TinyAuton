"""Check built local links, assets and preserved heading anchors.

Usage: python DOC/tools/check_site.py PATH_TO_BUILT_SITE
External URLs are deliberately not fetched.
"""
from html.parser import HTMLParser
import argparse
import json
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.references = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.references.append(attrs["href"])
        if tag in ("img", "script") and "src" in attrs:
            self.references.append(attrs["src"])
        if tag == "link" and attrs.get("rel") in ("stylesheet", "icon"):
            self.references.append(attrs.get("href", ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    parser.add_argument("--base-path", default="/", help="Deployment prefix, such as /TinyAuton/.")
    args = parser.parse_args()
    site = args.site.resolve()
    base_path = "/" + args.base_path.strip("/") + "/" if args.base_path.strip("/") else "/"
    if not (site / "index.html").is_file():
        print(f"Not a built site: {site}", file=sys.stderr)
        return 2
    pages = {}
    for path in site.rglob("*.html"):
        page = Page()
        page.feed(path.read_text(encoding="utf-8"))
        pages[path.resolve()] = page
    errors = set()
    checked = 0
    for path, page in pages.items():
        for reference in page.references:
            url = urlsplit(reference)
            if url.scheme or url.netloc or not reference:
                continue
            url_path = unquote(url.path)
            if base_path != "/" and url_path.startswith(base_path):
                url_path = "/" + url_path[len(base_path):]
            target = (site / url_path.lstrip("/")) if url_path.startswith("/") else (path.parent / url_path)
            target = target.resolve() if url.path else path
            if target.is_dir():
                target = target / "index.html"
            checked += 1
            if not target.is_file():
                errors.add(f"Missing target: {path.relative_to(site)} -> {reference}")
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                errors.add(f"Missing fragment: {path.relative_to(site)} -> {reference}")
    baseline = json.loads(Path(__file__).with_name("preservation_baseline.json").read_text(encoding="utf-8"))
    for route, anchors in baseline.get("anchors", {}).items():
        path = (site / route).resolve()
        if path not in pages:
            errors.add(f"Missing original route: {route}")
            continue
        for anchor in set(anchors) - pages[path].ids:
            errors.add(f"Missing original anchor: {route}#{anchor}")
    if errors:
        print("\n".join(sorted(errors)), file=sys.stderr)
        return 1
    print(f"Checked {len(pages)} HTML pages, {checked} local references and "
          f"{sum(map(len, baseline.get('anchors', {}).values()))} original heading anchors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

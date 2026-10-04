"""Check language pairs, all Markdown fences and active navigation targets."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def main():
    errors = []
    pages = list(DOCS.rglob("*.md"))
    for page in pages:
        relative = page.relative_to(DOCS)
        for source, target in ((".en.md", ".zh.md"), (".zh.md", ".en.md")):
            if page.name.endswith(source) and not page.with_name(page.name[:-len(source)] + target).is_file():
                errors.append(f"Missing language pair: {relative}")
        marker = None
        opening = 0
        for number, line in enumerate(page.read_text(encoding="utf-8-sig").splitlines(), 1):
            match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
            if not match:
                continue
            if marker is None:
                marker, opening = match.group(1), number
            elif not match.group(2).strip() and match.group(1)[0] == marker[0] and len(match.group(1)) >= len(marker):
                marker = None
        if marker is not None:
            errors.append(f"Unclosed fence: {relative}:{opening}")
    nav = (ROOT / "mkdocs.yml").read_text(encoding="utf-8").split("\nnav:\n", 1)[1]
    targets = []
    for line in nav.splitlines():
        if line.lstrip().startswith("#"):
            continue
        for target in re.findall(r'"([^"\n]+\.md)"', line):
            targets.append(target)
            path = DOCS / target
            if not path.is_file() and not path.with_name(path.stem + ".en.md").is_file():
                errors.append(f"Missing navigation page: {target}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Checked {len(pages)} Markdown pages, language pairs, fences and {len(targets)} navigation targets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

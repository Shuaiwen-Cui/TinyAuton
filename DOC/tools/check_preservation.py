"""Check historical Markdown fences and original documentation assets.

Run from any directory: python DOC/tools/check_preservation.py
The baseline predates the documentation reorganization. Body hashes are per
page multisets: moving or folding a block is fine; deletion or editing fails.
Line endings are normalized so Git CRLF/LF conversion is harmless.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).with_name("preservation_baseline.json")


def fence_hashes(path):
    lines = path.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    result = Counter()
    i = 0
    while i < len(lines):
        match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", lines[i])
        if not match:
            i += 1
            continue
        marker = match.group(1)
        end = i + 1
        closing = r"^\s*" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}\s*$"
        while end < len(lines) and not re.match(closing, lines[end]):
            end += 1
        if end == len(lines):
            raise ValueError(f"Unclosed fence: {path}:{i + 1}")
        digest = hashlib.sha256("".join(lines[i + 1:end]).encode("utf-8")).hexdigest()
        result[digest] += 1
        i = end + 1
    return result


def main():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    errors = []
    for name, hashes in baseline["pages"].items():
        path = ROOT / name
        if not path.is_file():
            errors.append(f"Missing page: {name}")
            continue
        try:
            missing = Counter(hashes) - fence_hashes(path)
            if missing:
                errors.append(f"Changed/missing blocks: {name}: {dict(missing)}")
        except ValueError as error:
            errors.append(str(error))
    for item in baseline["assets"]:
        path = ROOT / item["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            errors.append(f"Changed/missing asset: {item['path']}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Preserved {sum(map(len, baseline['pages'].values()))} historical blocks, "
          f"{len(baseline['pages'])} original pages and {len(baseline['assets'])} assets.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

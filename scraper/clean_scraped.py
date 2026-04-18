"""
Clean scraped Telecom Egypt markdown files:
- Remove repeated header (nav menus, icon sprites, logo links)
- Remove repeated footer (compare section, site footer, copyright, complementary content)
- Remove markdown image lines
- Detect and report duplicate files by content hash
- Output cleaned files to data/cleaned/
- Delete duplicate files, keeping only one per group
"""

import re
import hashlib
import json
from pathlib import Path

INPUT_DIR = Path("data/scraped")
OUTPUT_DIR = Path("data/cleaned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Known nav link texts that appear in the site-wide header nav
NAV_LINK_TEXTS = {
    "Personal", "Business", "Corporate Sustainability", "Investor Relations", "CSR",
    "My WE", "Manage My Landline", "Manage My Internet", "Manage My Mobile",
    "Log In", "5G", "Mobile", "Home", "Services", "Devices", "Promotions",
    "Shop", "WE Pay", "Sustainability", "Climate change", "Corporate Quality",
    "Mobile Services", "Data & Connectivity", "Voice Services",
    "Hosting & Data Center", "Digital Solutions", "Wholesale",
    "About Us", "Corporate Strategy", "Media Center", "Careers and Training", "Contact Us",
}


def is_nav_line(line: str) -> bool:
    """Return True if this line is part of the site-wide navigation boilerplate."""
    stripped = line.strip()

    # Icon sprite line
    if re.match(r'^arrowcaller.*logo\s*$', stripped):
        return True

    # Image-only lines (inline images with no surrounding text)
    if re.match(r'^\[?\s*!\[.*?\]\(.*?\)\s*\]?\(.*?\)\s*$', stripped):
        return True
    if re.match(r'^!\[.*?\]\(.*?\)\s*$', stripped):
        return True

    # Lines like: [My WE](...) or [Manage My Landline](...) etc.
    m = re.match(r'^\[([^\]]+)\]\(https?://[^\)]+\)\s*$', stripped)
    if m and m.group(1).strip() in NAV_LINK_TEXTS:
        return True

    # Lines like: * [ Personal ](url) or * [5G ](url)
    m = re.match(r'^\*\s*\[?\s*([^\]\[]+?)\s*\]?\s*\(https?://[^\)]+\)\s*(?:!\[.*?\]\(.*?\))?\s*$', stripped)
    if m and m.group(1).strip() in NAV_LINK_TEXTS:
        return True

    # Empty link lines: * [](url) or [ ](url)
    if re.match(r'^\*?\s*\[?\s*\]?\s*\(https?://[^\)]+\)\s*$', stripped):
        return True

    # IBM Logo line
    if 'IBM Logo' in stripped and '![' in stripped:
        return True

    return False


def extract_content(text: str) -> str:
    """Strip header and footer boilerplate, remove images."""

    # --- Strip footer first (easier boundary) ---
    footer_markers = [
        "\nCompare \n", "\nCompare\n",
        "\nTELECOMEGYPT\n",
        "\nComplementary Content",
        "\nCopyright ©",
        "\n##### You're using an unsupported browser",
    ]
    for marker in footer_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]

    # --- Find where actual content starts ---
    # Strategy 1: last "## <Something>" heading that is NOT "Web Content Viewer"
    # The nav ends with a "## Web Content Viewer" or "## StoresFinder" etc. heading
    # We want everything AFTER that heading line.
    wcv_pattern = re.compile(r'\n##\s+\S.*\n', re.MULTILINE)
    last_match = None
    for m in wcv_pattern.finditer(text):
        last_match = m
    if last_match:
        text = text[last_match.end():]
    else:
        # Strategy 2: strip line-by-line nav until we hit real content
        lines = text.splitlines()
        start = 0
        for i, line in enumerate(lines):
            if not is_nav_line(line) and line.strip() and not line.strip().startswith('*'):
                # Check if this looks like real content (not a nav section header like "Mobile Tariffs")
                start = i
                break
        text = "\n".join(lines[start:])

    # --- Remove image lines ---
    lines = text.splitlines()
    lines = [l for l in lines if not re.match(r'^\s*!\[.*?\]\(.*?\)\s*$', l)]

    # --- Remove pure nav list lines that leaked through ---
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip empty link-only lines
        if re.match(r'^\*\s*\[?\s*\]?\s*\(https?://[^\)]+\)\s*$', stripped):
            continue
        # Skip portlet JSON state lines
        if re.match(r'^\{\"Z\d', stripped):
            continue
        # Skip lines like: Z7_XXXXX (portlet IDs)
        if re.match(r'^Z\d_[A-Z0-9]+$', stripped):
            continue
        # Skip /wps/contenthandler lines
        if stripped.startswith('/wps/contenthandler/'):
            continue
        cleaned.append(line)

    content = "\n".join(cleaned)

    # Collapse 3+ consecutive blank lines into 2
    content = re.sub(r'\n{3,}', '\n\n', content)

    return content.strip()


def file_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def main():
    # First pass: collect all cleaned content and detect duplicates
    file_data: list[tuple[Path, str, str, str]] = []  # (path, frontmatter, cleaned_body, hash)
    stats = {"total": 0, "skipped": 0, "duplicates_removed": 0, "written": 0}

    for md_file in sorted(INPUT_DIR.glob("*.md")):
        stats["total"] += 1
        raw = md_file.read_text(encoding="utf-8", errors="ignore")

        # Parse frontmatter
        frontmatter = ""
        body = raw
        if raw.startswith("---"):
            end = raw.find("---", 3)
            if end != -1:
                frontmatter = raw[:end + 3]
                body = raw[end + 3:]

        cleaned_body = extract_content(body)

        if not cleaned_body.strip():
            stats["skipped"] += 1
            continue

        h = file_hash(cleaned_body)
        file_data.append((md_file, frontmatter, cleaned_body, h))

    # Group by hash to find duplicates
    hash_groups: dict[str, list[tuple[Path, str, str]]] = {}
    for md_file, fm, body, h in file_data:
        hash_groups.setdefault(h, []).append((md_file, fm, body))

    dup_report = []
    seen_hashes: set[str] = set()

    # Clear output dir of old .md files
    for old in OUTPUT_DIR.glob("*.md"):
        old.unlink()

    for h, group in hash_groups.items():
        if len(group) > 1:
            # Keep the file with the shortest/simplest name (prefer human-readable names)
            group_sorted = sorted(group, key=lambda x: (len(x[0].stem), x[0].name))
            keeper = group_sorted[0]
            duplicates = [g[0].name for g in group_sorted[1:]]
            dup_report.append({
                "hash": h,
                "kept": keeper[0].name,
                "removed": duplicates,
            })
            stats["duplicates_removed"] += len(duplicates)
            print(f"  Keeping '{keeper[0].name}', removing {len(duplicates)} duplicate(s): {', '.join(duplicates[:3])}{'...' if len(duplicates) > 3 else ''}")
            group = [keeper]

        md_file, fm, body = group[0]
        out_path = OUTPUT_DIR / md_file.name
        out_path.write_text(fm + "\n" + body + "\n", encoding="utf-8")
        stats["written"] += 1

    print(f"\nStats: {stats['total']} input, {stats['skipped']} skipped (empty), "
          f"{stats['duplicates_removed']} duplicates removed, {stats['written']} files written")

    report_path = OUTPUT_DIR / "_duplicates_report.json"
    report_path.write_text(json.dumps(dup_report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Duplicate report saved to {report_path}")
    print(f"Cleaned files saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

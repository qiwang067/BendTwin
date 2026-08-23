#!/usr/bin/env python3
import argparse
import os
import re
import shutil
from pathlib import Path


ROOT = Path("/code/phystwinv2_explicit")
RESULTS = ROOT / "results"
ROOT_TYPES = (
    "gaussian_output",
    "gaussian_output_dynamic",
    "gaussian_output_dynamic_white",
)
EXCLUDED_PARENT_PREFIXES = (
    "gaussian_output_video",
    "gaussian_output_reconstruction",
)


def parent_kind_and_suffix(parent: str):
    for kind in ("gaussian_output_dynamic_white", "gaussian_output_dynamic", "gaussian_output"):
        if parent == kind:
            return kind, ""
        if parent.startswith(kind + "_"):
            return kind, parent[len(kind) + 1 :]
    return None, None


def is_hash_name(name: str) -> bool:
    return name.startswith("mat") or name.startswith("mate")


def is_allowed_hash_dir(path: Path) -> bool:
    try:
        rel = path.relative_to(ROOT)
    except ValueError:
        return False
    if len(rel.parts) != 2:
        return False
    parent, name = rel.parts
    if not is_hash_name(name):
        return False
    if any(parent.startswith(prefix) for prefix in EXCLUDED_PARENT_PREFIXES):
        return False
    kind, _suffix = parent_kind_and_suffix(parent)
    return kind in ROOT_TYPES


def counterpart_dirs(hash_dir: Path) -> set[Path]:
    if not is_allowed_hash_dir(hash_dir):
        return set()
    parent = hash_dir.parent.name
    hash_name = hash_dir.name
    _kind, suffix = parent_kind_and_suffix(parent)

    dirs = set()
    for kind in ROOT_TYPES:
        parent_name = f"{kind}_{suffix}" if suffix else kind
        candidate = ROOT / parent_name / hash_name
        if candidate.exists() and is_allowed_hash_dir(candidate):
            dirs.add(candidate)
    return dirs


def candidate_tokens(token: str) -> set[str]:
    token = token.strip().split(",", 1)[0]
    tokens = {token}
    for match in re.finditer(r"(mat[a-zA-Z0-9]{8,}|mate[a-zA-Z0-9]{8,})", token):
        tokens.add(match.group(1))
    return tokens


def collect_nonbest_hash_dirs():
    all_candidate_dirs: set[Path] = set()
    selected_dirs: set[Path] = set()
    unresolved_selected: list[tuple[Path, str]] = []
    per_summary = []

    for summary_path in sorted(RESULTS.glob("best_vs_pt*_summary.txt")):
        lines = summary_path.read_text(encoding="utf-8", errors="replace").splitlines()
        token_to_dirs: dict[str, set[Path]] = {}
        current_candidate = None
        current_hash = None
        local_candidates: set[Path] = set()

        for line in lines:
            if line.startswith("Candidate "):
                current_candidate = line.split("Candidate ", 1)[1].strip()
                current_hash = None
                for token in candidate_tokens(current_candidate):
                    token_to_dirs.setdefault(token, set())
                continue

            list_candidate = re.match(r"\s*-\s+(\S+)$", line)
            if list_candidate and (
                "ssr" in list_candidate.group(1) or list_candidate.group(1).startswith("mat")
            ):
                current_candidate = list_candidate.group(1).strip()
                current_hash = None
                for token in candidate_tokens(current_candidate):
                    token_to_dirs.setdefault(token, set())
                continue

            hash_match = re.match(r"\s*Hash:\s*(\S+)", line) or re.match(r"Hash\s+(\S+)", line)
            if hash_match:
                current_hash = hash_match.group(1).strip().strip(",")
                token_to_dirs.setdefault(current_hash, set())
                continue

            path_match = re.search(r"Material config path:\s*(/\S+/material_config\.json)", line)
            if path_match:
                base_hash_dir = Path(path_match.group(1)).parent
                dirs = counterpart_dirs(base_hash_dir)
                if dirs:
                    local_candidates.update(dirs)
                    all_candidate_dirs.update(dirs)
                    keys = {base_hash_dir.name}
                    if current_candidate:
                        keys.add(current_candidate)
                        keys.update(candidate_tokens(current_candidate))
                    if current_hash:
                        keys.add(current_hash)
                    for key in keys:
                        token_to_dirs.setdefault(key, set()).update(dirs)

        in_selected = False
        local_selected_tokens = []
        local_selected_dirs: set[Path] = set()
        for line in lines:
            if line.startswith("Selected scenes after cross-hash merge"):
                in_selected = True
                continue
            if in_selected and line.startswith("Scenes not beaten by any hash"):
                in_selected = False
            if not in_selected:
                continue

            selected_match = re.match(r"\s*-\s+[^:]+:\s*([^,]+),", line)
            if not selected_match:
                continue
            token = selected_match.group(1).strip()
            local_selected_tokens.append(token)
            dirs = set()
            for candidate_token in candidate_tokens(token):
                dirs.update(token_to_dirs.get(candidate_token, set()))
            if dirs:
                selected_dirs.update(dirs)
                local_selected_dirs.update(dirs)
            else:
                unresolved_selected.append((summary_path, token))

        per_summary.append(
            {
                "path": summary_path,
                "candidate_count": len(local_candidates),
                "protected_count": len(local_selected_dirs),
                "selected_rows": len(local_selected_tokens),
            }
        )

    return {
        "summary_count": len(list(RESULTS.glob("best_vs_pt*_summary.txt"))),
        "candidate_dirs": all_candidate_dirs,
        "protected_dirs": selected_dirs,
        "delete_dirs": sorted(all_candidate_dirs - selected_dirs),
        "unresolved_selected": unresolved_selected,
        "per_summary": per_summary,
    }


def collect_core_files() -> list[Path]:
    return sorted(path for path in ROOT.glob("core.*") if path.is_file())


def path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size

    total = 0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            file_path = Path(root) / filename
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def fmt_size(size: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def delete_path(path: Path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()


def main():
    parser = argparse.ArgumentParser(
        description="Clean non-selected material hash outputs and top-level core.* files."
    )
    parser.add_argument("--delete", action="store_true", help="Actually delete listed paths.")
    parser.add_argument(
        "--no-core",
        action="store_true",
        help="Do not include top-level /code/phystwinv2_explicit/core.* files.",
    )
    args = parser.parse_args()

    result = collect_nonbest_hash_dirs()
    delete_dirs = result["delete_dirs"]
    core_files = [] if args.no_core else collect_core_files()
    delete_paths = delete_dirs + core_files
    sizes = [(path, path_size(path)) for path in delete_paths]
    total_size = sum(size for _path, size in sizes)

    print("DELETE MODE" if args.delete else "DRY RUN ONLY - no files were deleted")
    print("Included output roots: gaussian_output, gaussian_output_dynamic, gaussian_output_dynamic_white")
    print(f"Summary files scanned: {result['summary_count']}")
    print(f"Allowed candidate/counterpart hash dirs found: {len(result['candidate_dirs'])}")
    print(f"Selected-best hash dirs protected: {len(result['protected_dirs'])}")
    print(f"Hash dirs marked for deletion: {len(delete_dirs)}")
    print(f"Core files marked for deletion: {len(core_files)}")
    print(f"Total size marked for deletion: {fmt_size(total_size)}")
    print()

    print("Per-summary candidate/protected counts:")
    for item in result["per_summary"]:
        print(
            f"  {item['path'].name}: "
            f"candidates={item['candidate_count']}, "
            f"protected={item['protected_count']}, "
            f"selected_rows={item['selected_rows']}"
        )
    print()

    if result["unresolved_selected"]:
        print("Unresolved selected tokens:")
        seen = set()
        for summary_path, token in result["unresolved_selected"]:
            key = (summary_path.name, token)
            if key in seen:
                continue
            seen.add(key)
            print(f"  {summary_path.name}: {token}")
        print()

    print("Paths marked for deletion:")
    for path, size in sizes:
        print(f"  {fmt_size(size):>10}  {path}")

    if args.delete:
        for path, _size in sizes:
            delete_path(path)
        print()
        print(f"Deleted {len(sizes)} paths.")


if __name__ == "__main__":
    main()

"""Check that the files about to be published are the files that were verified.

The release workflow builds wheels, verifies each one on a runner of its own
platform, and then uploads. Between those steps the upload directory is
assembled from workflow artifacts, and that assembly is where a release can go
wrong without anything failing:

* the release workflow calls the full CI workflow, which uploads wheels of its
  own, and a `download-artifact` step with no name or pattern downloads *every*
  artifact in the run;
* `merge-multiple: true` then puts them all in one directory, where a CI wheel
  and a release wheel for the same platform have the *same filename* -- so one
  silently replaces the other;
* the surviving file may be a build that no verification job ever ran.

So each verify job records the sha256 of the artifact it actually installed
and ran, and this compares the upload directory against the union of those
records. Anything unexpected, missing, duplicated, or changed stops the
release.

    python scripts/check_release_set.py --dist dist --manifests manifests
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifests(folder: pathlib.Path) -> tuple[dict, list[str]]:
    """Every `*.json` under `folder`, keyed by filename.

    Each verify job writes one. A filename appearing in two manifests means
    two different jobs verified two different files that would land on top of
    each other, which is the collision this exists to catch.
    """
    verified: dict[str, dict] = {}
    problems: list[str] = []
    files = sorted(folder.rglob("*.json"))
    if not files:
        problems.append(f"no verification manifests found under {folder}")
    for path in files:
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{path.name} is not readable JSON: {exc}")
            continue
        name = entry.get("filename")
        if not name or not entry.get("sha256"):
            problems.append(f"{path} is missing filename/sha256")
            continue
        if name in verified and verified[name]["sha256"] != entry["sha256"]:
            problems.append(
                f"two verification jobs recorded different content for the "
                f"same filename {name}: {verified[name]['sha256'][:16]} "
                f"(from {verified[name]['job']}) and {entry['sha256'][:16]} "
                f"(from {entry.get('job')})")
        verified[name] = entry
    return verified, problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=pathlib.Path, required=True,
                    help="the directory that is about to be uploaded")
    ap.add_argument("--manifests", type=pathlib.Path, required=True,
                    help="directory of manifests written by the verify jobs")
    ap.add_argument("--expect", type=int, default=None,
                    help="exact number of files the release must contain")
    args = ap.parse_args()

    verified, problems = load_manifests(args.manifests)

    present = sorted(p for p in args.dist.rglob("*") if p.is_file())
    distributions = [p for p in present
                     if p.suffix == ".whl" or p.name.endswith(".tar.gz")]
    strays = [p for p in present if p not in distributions]
    if strays:
        problems.append(
            "the upload directory holds files that are not distributions: "
            + ", ".join(sorted(p.name for p in strays)))

    # A filename appearing twice under different paths is the CI-wheel
    # collision: merge-multiple would have flattened them onto each other.
    by_name: dict[str, list[pathlib.Path]] = {}
    for path in distributions:
        by_name.setdefault(path.name, []).append(path)
    for name, paths in sorted(by_name.items()):
        if len(paths) > 1:
            digests = {sha256(p) for p in paths}
            problems.append(
                f"{name} appears {len(paths)} times "
                f"({', '.join(str(p) for p in paths)}) with "
                f"{len(digests)} distinct contents")

    for name in sorted(by_name):
        if name not in verified:
            problems.append(
                f"{name} is in the upload directory but no verification job "
                "recorded it. Only artifacts that were installed and run may "
                "be published.")

    for name, entry in sorted(verified.items()):
        if name not in by_name:
            problems.append(
                f"{name} was verified by {entry.get('job')} but is absent "
                "from the upload directory")
            continue
        actual = sha256(by_name[name][0])
        if actual != entry["sha256"]:
            problems.append(
                f"{name} changed after it was verified: "
                f"{entry['sha256'][:16]}... became {actual[:16]}...")

    if args.expect is not None and len(by_name) != args.expect:
        problems.append(
            f"expected exactly {args.expect} distributions, found "
            f"{len(by_name)}: {sorted(by_name)}")

    print(f"upload directory: {args.dist}")
    for name in sorted(by_name):
        entry = verified.get(name, {})
        print(f"  {name}")
        print(f"    sha256 {sha256(by_name[name][0])}")
        print(f"    verified by {entry.get('job', 'NOTHING')} "
              f"on {entry.get('runner', '?')}")

    if problems:
        print("\nthe release set does not match what was verified:",
              file=sys.stderr)
        for line in problems:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print(f"\n{len(by_name)} distributions, each matching the artifact a "
          "verification job installed and ran")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

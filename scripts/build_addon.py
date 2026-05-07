"""Build the blender-pixeloe addon zip with scipy wheels bundled.

Pulls scipy wheels for each target platform from PyPI, writes the final
blender_manifest.toml with the bundled wheel filenames, and packs the addon
zip into dist/.

Run from the repo root:
    uv run python scripts/build_addon.py

The output is dist/blender_pixeloe-<version>.zip, ready to install in Blender
via Edit -> Preferences -> Get Extensions -> dropdown -> Install from Disk.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ADDON_DIR = REPO_ROOT / "blender_pixeloe"
MANIFEST_PATH = ADDON_DIR / "blender_manifest.toml"
WHEELS_DIR = ADDON_DIR / "wheels"
DIST_DIR = REPO_ROOT / "dist"

SCIPY_VERSION = "1.16.3"

# Blender ships its own Python and the version moves between releases:
#   4.2 LTS .. 5.0  -> 3.11
#   5.1+            -> 3.14
# We bundle wheels for every ABI we want the addon to install on. Blender
# filters by ABI tag at install time and picks the matching wheel; users on
# 4.2 get the cp311 wheel, users on 5.1 get the cp314 wheel.
PYTHON_TAGS = ["311", "314"]

# Maps the manifest's `platforms` value to the pip --platform tag(s) that pull
# the right wheel from PyPI. Multiple tags per platform handle pip's strict
# tag matching (e.g. macOS arm64 wheels publish as macosx_12_0_arm64 but we
# also try macosx_11_0_arm64 in case scipy bumps the floor).
PLATFORM_PIP_TAGS: dict[str, list[str]] = {
    "linux-x64": ["manylinux2014_x86_64"],
    "windows-x64": ["win_amd64"],
    "macos-x64": ["macosx_10_9_x86_64", "macosx_11_0_x86_64", "macosx_12_0_x86_64"],
    "macos-arm64": ["macosx_11_0_arm64", "macosx_12_0_arm64", "macosx_13_0_arm64"],
}


def fetch_wheels(dest: Path) -> list[Path]:
    """Download scipy wheels for every (platform, python ABI) pair into `dest`."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    fetched: list[Path] = []
    for python_tag in PYTHON_TAGS:
        for platform, pip_tags in PLATFORM_PIP_TAGS.items():
            wheel = _download_one(platform, python_tag, pip_tags, dest)
            fetched.append(wheel)
    return fetched


def _download_one(
    platform: str, python_tag: str, pip_tags: list[str], dest: Path
) -> Path:
    """Try each pip tag for `platform` until one succeeds; return the wheel path."""
    last_err: subprocess.CalledProcessError | None = None
    existing = {p.name for p in dest.glob("scipy-*.whl")}
    for tag in pip_tags:
        cmd = [
            sys.executable, "-m", "pip", "download",
            "--no-deps",
            "--only-binary=:all:",
            "--platform", tag,
            "--python-version", python_tag,
            "--implementation", "cp",
            "--abi", f"cp{python_tag}",
            "--dest", str(dest),
            f"scipy=={SCIPY_VERSION}",
        ]
        print(f"[cp{python_tag} {platform}] trying tag={tag}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            last_err = e
            continue

        new_files = {p.name for p in dest.glob("scipy-*.whl")} - existing
        for name in new_files:
            whl = dest / name
            if _matches(whl.name, platform, python_tag):
                print(f"[cp{python_tag} {platform}] -> {whl.name}")
                return whl
            whl.unlink()
    raise RuntimeError(
        f"Failed to fetch scipy wheel for cp{python_tag} {platform}. "
        f"Last pip stderr:\n{last_err.stderr if last_err else '(no error captured)'}"
    )


def _matches(wheel_name: str, manifest_platform: str, python_tag: str) -> bool:
    name = wheel_name.lower()
    if f"cp{python_tag}" not in name:
        return False
    return _matches_platform(name, manifest_platform)


def _matches_platform(wheel_name: str, manifest_platform: str) -> bool:
    """Check whether a downloaded wheel filename matches the manifest platform."""
    name = wheel_name.lower()
    if manifest_platform == "linux-x64":
        return "manylinux" in name and "x86_64" in name
    if manifest_platform == "windows-x64":
        return "win_amd64" in name
    if manifest_platform == "macos-x64":
        return "macosx" in name and "x86_64" in name
    if manifest_platform == "macos-arm64":
        return "macosx" in name and "arm64" in name
    return False


def read_addon_version() -> str:
    data = tomllib.loads(MANIFEST_PATH.read_text())
    return data["version"]


def pack_zip(version: str, wheels: list[Path]) -> Path:
    """Write the addon zip with manifest and source files at the zip root."""
    DIST_DIR.mkdir(exist_ok=True)
    zip_path = DIST_DIR / f"blender_pixeloe-{version}.zip"
    if zip_path.exists():
        zip_path.unlink()

    addon_files = [
        p for p in ADDON_DIR.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix not in {".pyc"}
        and p.name != "blender_manifest.toml"  # we substitute the rendered one
        and "wheels" not in p.parts  # bundled separately below
    ]

    rendered_manifest = _render_manifest(wheels)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("blender_manifest.toml", rendered_manifest)
        for f in addon_files:
            arcname = f.relative_to(ADDON_DIR).as_posix()
            zf.write(f, arcname=arcname)
        for whl in wheels:
            zf.write(whl, arcname=f"wheels/{whl.name}")

    return zip_path


def _render_manifest(wheels: list[Path]) -> str:
    """Same substitution logic as write_manifest_with_wheels but returning text."""
    src = MANIFEST_PATH.read_text()
    block = "wheels = [\n" + "\n".join(
        f'  "./wheels/{w.name}",' for w in wheels
    ) + "\n]"
    out_lines: list[str] = []
    in_wheels = False
    replaced = False
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("wheels = ["):
            if "]" in stripped:
                out_lines.append(block)
                replaced = True
                continue
            in_wheels = True
            continue
        if in_wheels:
            if stripped == "]":
                out_lines.append(block)
                in_wheels = False
                replaced = True
            continue
        out_lines.append(line)
    if not replaced:
        raise RuntimeError("Did not find a `wheels = [...]` block in the manifest.")
    return "\n".join(out_lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Reuse wheels already in blender_pixeloe/wheels/",
    )
    args = ap.parse_args()

    if args.skip_fetch:
        if not WHEELS_DIR.exists():
            print(f"--skip-fetch but {WHEELS_DIR} does not exist", file=sys.stderr)
            return 1
        wheels = sorted(WHEELS_DIR.glob("scipy-*.whl"))
        if not wheels:
            print(f"--skip-fetch but {WHEELS_DIR} is empty", file=sys.stderr)
            return 1
    else:
        wheels = fetch_wheels(WHEELS_DIR)

    version = read_addon_version()
    zip_path = pack_zip(version, wheels)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\nBuilt {zip_path} ({size_mb:.1f} MB)")
    print(f"Wheels bundled: {len(wheels)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

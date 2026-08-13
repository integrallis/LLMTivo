#!/usr/bin/env python3
"""Check for dependency conflicts in pyproject.toml minimum versions."""

import json
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def load_dependencies() -> dict[str, str]:
    """Load minimum dependency versions from pyproject.toml."""
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    deps = {}
    for dep in data["project"]["dependencies"]:
        match = re.match(r"([a-zA-Z0-9_-]+)>=([0-9.]+)", dep)
        if match:
            pkg, ver = match.groups()
            deps[pkg] = ver
    return deps


def fetch_package_metadata(pkg: str, ver: str) -> dict:
    """Fetch PyPI metadata for a package version."""
    url = f"https://pypi.org/pypi/{pkg}/{ver}/json"
    with urlopen(url, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def version_parts(version: str) -> list[int]:
    """Convert a dotted version string to comparable integer parts."""
    return [int(x) for x in version.split(".")]


def requires_higher_version(requirement: str, package: str, current_version: str) -> str | None:
    """Return the required version if a requirement exceeds the current minimum."""
    if not (requirement.startswith(package + ">") or requirement.startswith(package + " ")):
        return None

    match = re.search(r">=(\d+\.\d+(?:\.\d+)?)", requirement)
    if not match:
        return None

    required_version = match.group(1)
    required_parts = version_parts(required_version)
    current_parts = version_parts(current_version)

    while len(required_parts) < len(current_parts):
        required_parts.append(0)
    while len(current_parts) < len(required_parts):
        current_parts.append(0)

    return required_version if required_parts > current_parts else None


def find_conflicts(deps: dict[str, str]) -> list[str]:
    """Find dependency minimum-version conflicts."""
    conflicts = []
    for pkg, ver in deps.items():
        try:
            pkg_data = fetch_package_metadata(pkg, ver)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            continue

        requires = pkg_data.get("info", {}).get("requires_dist", [])
        for req in requires:
            if "extra ==" in req:
                continue
            for our_pkg, our_ver in deps.items():
                required_ver = requires_higher_version(req, our_pkg, our_ver)
                if required_ver:
                    conflicts.append(
                        f"  ❌ {pkg}=={ver} requires {our_pkg}>={required_ver}, but we have >={our_ver}"
                    )
    return conflicts


def main() -> int:
    """Check dependency minimum-version conflicts."""
    conflicts = find_conflicts(load_dependencies())
    if conflicts:
        print("\n⚠️  Dependency conflicts found:")
        for c in conflicts:
            print(c)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

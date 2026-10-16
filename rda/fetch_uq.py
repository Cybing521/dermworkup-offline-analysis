#!/usr/bin/env python3
"""Try to locate UQ eSpace files for Ghahari 2025. Does not download images."""
import json
import urllib.request

URLS = [
    "https://doi.org/10.48610/a13deaf",
    "https://espace.library.uq.edu.au/view/UQ:a13deaf",
    "https://api.datacite.org/dois/10.48610/a13deaf",
]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "RDA-dataset-locator/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return {"url": url, "status": r.status, "final": r.geturl(), "bytes": len(r.read(4096))}
    except Exception as e:
        return {"url": url, "error": str(e)}


if __name__ == "__main__":
    print(json.dumps([fetch(u) for u in URLS], indent=2))

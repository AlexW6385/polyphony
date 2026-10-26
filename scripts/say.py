#!/usr/bin/env python3
"""Drop a conscious thought into the mind.

    python scripts/say.py "should I take the AI specialization?"
    python scripts/say.py --world --author news "Some headline"
"""

import argparse
import json
import os
import sys
import urllib.request


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("content", nargs="+", help="the thought")
    p.add_argument("--world", action="store_true", help="inject as a world thought instead")
    p.add_argument("--author", default="world", help="author for --world thoughts")
    p.add_argument("--url", default=os.environ.get("POLYPHONY_URL", "http://localhost:8000"))
    a = p.parse_args()

    body = {"content": " ".join(a.content), "kind": "world" if a.world else "conscious"}
    if a.world:
        body["author"] = a.author
    req = urllib.request.Request(
        a.url.rstrip("/") + "/thoughts",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            t = json.load(r)
    except OSError as e:
        sys.exit(f"could not reach {a.url}: {e}")
    print(f"[{t['seq']}] {t['author']}: {t['content']}")


if __name__ == "__main__":
    main()

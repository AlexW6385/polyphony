#!/usr/bin/env python3
"""Watch the stream live in a terminal.

    python scripts/tail.py            # from now on
    python scripts/tail.py --all      # replay everything first
"""

import argparse
import json
import os
import sys
import textwrap
import urllib.request

COLORS = {"conscious": "\033[33m", "world": "\033[32m", "cadence": "\033[35m", "voice": "\033[36m"}
RESET, DIM = "\033[0m", "\033[2m"


def show(t: dict, color: bool) -> None:
    c, r, d = (COLORS.get(t["kind"], ""), RESET, DIM) if color else ("", "", "")
    mood = f" ({t['mood']})" if t.get("mood") else ""
    kind = "" if t["kind"] == "voice" else f" [{t['kind']}]"
    print(f"{d}[{t['seq']}]{r} {c}{t['author']}{r}{kind}{d}{mood}{r}")
    print(textwrap.indent(t["content"], "    "))
    print(flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--all", action="store_true", help="replay the whole stream first")
    p.add_argument("--url", default=os.environ.get("POLYPHONY_URL", "http://localhost:8000"))
    a = p.parse_args()
    color = sys.stdout.isatty()
    url = a.url.rstrip("/") + "/stream" + ("?after=0" if a.all else "")
    try:
        with urllib.request.urlopen(url) as r:
            for raw in r:
                line = raw.decode().rstrip("\n")
                if line.startswith("data: "):
                    show(json.loads(line[6:]), color)
    except KeyboardInterrupt:
        pass
    except OSError as e:
        sys.exit(f"could not reach {a.url}: {e}")


if __name__ == "__main__":
    main()

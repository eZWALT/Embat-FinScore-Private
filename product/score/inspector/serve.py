"""Local inspector for an export bundle: a tiny page to check that what the pipeline generated is sound, from the bundle alone.

    python product/score/inspector/serve.py --bundle <bundle_dir> [--port 8765] [--open]

Serves index.html and the bundle (as /bundle/...) on 127.0.0.1 only. Standard library only, nothing to install.
This is a developer tool for understanding and auditing the numbers, not the product (the product is product/web, built by the team).
"""
from __future__ import annotations

import argparse
import functools
import http.server
import sys
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, bundle: Path, **kw):
        self.bundle = bundle
        super().__init__(*a, directory=str(HERE), **kw)

    def translate_path(self, path: str) -> str:
        p = path.split("?", 1)[0]
        if p.startswith("/bundle/"):
            rel = Path(*[x for x in p[len("/bundle/"):].split("/") if x and x != ".."])
            return str(self.bundle / rel)
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):  # quiet
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = ap.parse_args()
    if not (args.bundle / "manifest.json").exists():
        print(f"no manifest.json in {args.bundle}", file=sys.stderr)
        return 1
    handler = functools.partial(Handler, bundle=args.bundle.resolve())
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as srv:
        url = f"http://127.0.0.1:{args.port}/"
        print(f"inspector on {url}  (bundle {args.bundle})  Ctrl+C to stop")
        if args.open:
            webbrowser.open(url)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

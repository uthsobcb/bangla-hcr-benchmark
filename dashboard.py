"""Live training dashboard at http://localhost:8765 — reads logs/*.log on each request.

    ./venv/bin/python dashboard.py          # then open the URL
    ./venv/bin/python dashboard.py --port 9000

ponytail: stdlib http.server + a <meta refresh>, no JS, no framework, no websockets.
The logs are the state; this just renders them.
"""
import argparse
import re
import subprocess
from datetime import datetime, timedelta
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).parent
LOGS = ROOT / "logs"
CKPTS = ROOT / "checkpoints"

# Two experiments, not one. Each row: (arch, log stem or None if trained before logging, checkpoint).
GROUPS = [
    ("Equal budget — 300 images per class",
     "the controlled 4-way comparison; ViT can't do full data in under 3 days",
     [("cnn", "cnn", "cnn_merged_lpc300.pth"),
      ("resnet18", "resnet18", "resnet18_merged_lpc300.pth"),
      ("extended_vit (warm-started)", "extended_vit", "extended_vit_merged_lpc300.pth"),
      ("extended_vit (ImageNet init)", "extended_vit_imagenet",
       "extended_vit_merged_lpc300_imagenet.pth"),
      ("vit", "vit", "vit_merged_lpc300.pth")]),
    ("Full data — all 681,309 images",
     "the headline numbers; pure ViT omitted (~70h)",
     [("cnn", "cnn_full", "cnn_merged.pth"),
      ("resnet18", None, "resnet18_merged.pth"),
      ("extended_vit", None, "extended_vit_merged.pth")]),
]
EPOCH_RE = re.compile(
    r"Epoch \[(\d+)/(\d+)\].*?Val Loss: ([\d.]+) Acc: ([\d.]+) \| Time: ([\d.]+)s \| Best Val: ([\d.]+)")
TEST_RE = re.compile(r"Test Accuracy: ([\d.]+)")


def read_run(name):
    log = LOGS / f"{name}.log"
    if not log.exists():
        return {"name": name, "state": "queued"}

    text = log.read_text(errors="replace")
    epochs = EPOCH_RE.findall(text)
    test = TEST_RE.findall(text)
    run = {"name": name, "state": "running", "epochs": len(epochs)}

    if epochs:
        cur, total, _, val_acc, secs, best = epochs[-1]
        run.update(epoch=int(cur), total=int(total), val_acc=float(val_acc),
                   secs=float(secs), best=float(best))
        remaining = int(total) - int(cur)
        if remaining > 0:
            run["eta"] = (datetime.now() + timedelta(seconds=remaining * float(secs))).strftime("%H:%M")
    if test:
        run.update(state="done", test_acc=float(test[-1]))
    return run


def running_arch():
    """Which arch the live process is on, read off its command line.

    ponytail: `ps ax -o args=`, not `pgrep -a` — macOS pgrep prints bare PIDs and ignores -a,
    so the command line (and the --arch we need) never comes back.
    """
    try:
        out = subprocess.run(["ps", "ax", "-o", "args="], capture_output=True, text=True).stdout
    except OSError:
        return None
    for line in out.splitlines():
        if "train_arch.py" not in line:
            continue
        m = re.search(r"--arch (\w+)", line)
        if not m:
            continue
        arch = m.group(1)
        if "--no-warm-start" in line:
            return f"{arch}_imagenet"
        return arch if "--limit-per-class" in line else f"{arch}_full"
    return None


def bar(frac, done):
    pct = max(0, min(100, frac * 100))
    cls = "done" if done else "live"
    return f'<div class="bar"><i class="{cls}" style="width:{pct:.1f}%"></i></div>'


_val_cache = {}


def best_val(ckpt_path):
    """best_val_acc recorded in the matching *_last.pth. Cached by mtime — these are 100-300MB."""
    last = ckpt_path.with_name(ckpt_path.stem + "_last.pth")
    if not last.exists():
        return None
    key = (str(last), last.stat().st_mtime)
    if key not in _val_cache:
        import torch  # lazy: keeps dashboard startup instant when nothing needs loading
        _val_cache.clear()
        _val_cache[key] = torch.load(last, map_location="cpu",
                                     weights_only=False).get("best_val_acc")
    return _val_cache[key]


def row_html(arch, log, ckpt, live):
    """One table row. A row with no log but an existing checkpoint was trained earlier."""
    path = CKPTS / ckpt
    if log is None:
        if not path.exists():
            return f'<tr class="q"><td>{escape(arch)}</td><td colspan="5">not trained</td></tr>'
        when = datetime.fromtimestamp(path.stat().st_mtime).strftime("%b %d")
        acc = best_val(path)
        acc_cell = f"{acc:.4f}<small> best val</small>" if acc else "—"
        return (f'<tr class="done"><td><b>{escape(arch)}</b></td><td>trained {when}</td>'
                f'<td>{bar(1, True)}</td><td class="n">{acc_cell}</td>'
                f'<td class="n">—</td><td class="n">—</td></tr>')

    r = read_run(log)
    if r["state"] == "running" and log != live and "epoch" not in r:
        r["state"] = "queued"
    if r["state"] == "running" and log != live and r.get("epoch") == r.get("total"):
        r["state"] = "done"

    if r["state"] == "queued":
        return f'<tr class="q"><td>{escape(arch)}</td><td colspan="5">queued</td></tr>'

    ep, tot = r.get("epoch", 0), r.get("total", 8)
    acc = r.get("test_acc", r.get("best", 0))
    acc_label = "test" if "test_acc" in r else "best val"
    state = "done" if r["state"] == "done" else f"epoch {ep}/{tot}"
    eta = "—" if r["state"] == "done" else r.get("eta", "—")
    return (f'<tr class="{r["state"]}"><td><b>{escape(arch)}</b></td><td>{state}</td>'
            f'<td>{bar(ep / tot if tot else 0, r["state"] == "done")}</td>'
            f'<td class="n">{acc:.4f}<small> {acc_label}</small></td>'
            f'<td class="n">{r.get("secs", 0):.0f}s<small>/ep</small></td>'
            f'<td class="n">{eta}</td></tr>')


def render():
    live = running_arch()
    blocks = []
    for title, note, runs in GROUPS:
        rows = "".join(row_html(a, lg, ck, live) for a, lg, ck in runs)
        blocks.append(f'<h2>{escape(title)}</h2><p class="note">{escape(note)}</p>'
                      f"<table>{rows}</table>")
    rows = "".join(blocks)

    status = f"running <b>{escape(live)}</b>" if live else "idle — nothing training"
    return f"""<!doctype html><meta charset=utf-8>
<meta http-equiv="refresh" content="20">
<title>bornomal — training</title>
<style>
:root{{color-scheme:dark light}}
body{{font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:#0d1117;color:#c9d1d9;
     margin:0;padding:22px}}
h1{{font-size:15px;margin:0 0 4px;color:#f0f6fc;letter-spacing:.02em}}
h2{{font-size:13px;margin:26px 0 2px;color:#f0f6fc}}
p.sub{{margin:0 0 18px;color:#8b949e;font-size:12px}}
p.note{{margin:0 0 8px;color:#6e7681;font-size:11px}}
table{{border-collapse:collapse;width:100%;max-width:760px}}
td{{padding:9px 10px;border-bottom:1px solid #21262d;vertical-align:middle}}
tr.q td{{color:#6e7681}}
tr.done td{{color:#7ee787}}
td.n{{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}}
small{{color:#6e7681;margin-left:3px}}
.bar{{background:#21262d;border-radius:3px;height:7px;width:150px;overflow:hidden}}
.bar i{{display:block;height:100%;background:#58a6ff}}
.bar i.done{{background:#3fb950}}
</style>
<h1>bornomal-pm · 4-method training</h1>
<p class="sub">{status} · updated {datetime.now():%H:%M:%S} · refreshes every 20s</p>
{rows}"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = render().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass  # a request line every 20s is just noise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    print(f"Dashboard: http://localhost:{args.port}  (ctrl-c to stop)")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()

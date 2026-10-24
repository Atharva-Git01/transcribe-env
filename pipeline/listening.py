"""The owner's blind listening check (PLAN_v0.9 section 5, stage 3): which version of a line is what you hear?

  trx screen listen --candidate B [--base A] [--n 20]     write evals/compare/listening_<set>_<A>_vs_<B>.html
  trx screen listen-score --answers <downloaded json>     count the owner's choices

The page is one local file (audio embedded, no internet, no outside scripts) in evals/compare/, which never leaves
this PC: it holds the private recordings. It shows N lines where the two candidates' outputs differ, each with its
audio and the two versions in a random order, unlabelled. The owner picks "1", "2", "same" or "cannot tell" and saves
the answers; the file names only line ids and choices. Transcripts have habits and gaps; the ear decides "as spoken".
"""
from __future__ import annotations

import base64
import html
import json
import random

from .common import TRX_HOME, TrxError, read_json, utc_now

COMPARE = TRX_HOME / "evals" / "compare"
SCREENING = TRX_HOME / "evals" / "screening"


def _wav_b64(path) -> str:
    """The clip as a 16-bit WAV data URI (the screening clips already are 16 kHz mono WAV)."""
    return "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def page(candidate: str, base: str = "indic_transcribe", set_id: str = "v1", n: int = 20, seed: int = 7) -> dict:
    d = SCREENING / set_id
    lines = {x["id"]: x for x in read_json(d / "lines.json")["segments"]}
    outs = {}
    for c in (base, candidate):
        f = d / "out" / f"{c}.json"
        if not f.is_file():
            raise TrxError("USAGE", f"no stored output of '{c}' on set {set_id}")
        outs[c] = {s["line"]: s["text"] for s in read_json(f)["segments"]}
    differ = sorted(i for i in lines if outs[base].get(i, "") != outs[candidate].get(i, "")
                    and outs[base].get(i) and outs[candidate].get(i))
    rnd = random.Random(seed)
    chosen = sorted(rnd.sample(differ, min(n, len(differ))), key=lambda i: (lines[i]["fixture"], lines[i]["start"]))
    key, items = {}, []
    for k, lid in enumerate(chosen, 1):
        order = [base, candidate]
        rnd.shuffle(order)
        key[lid] = order                      # which candidate is version 1 and 2 (kept out of the page)
        items.append((k, lid, [outs[c][lid] for c in order]))
    name = f"listening_{set_id}_{base}_vs_{candidate}"
    COMPARE.mkdir(parents=True, exist_ok=True)
    (COMPARE / f"{name}.key.json").write_text(json.dumps({"set": set_id, "base": base, "candidate": candidate,
                                                          "made_utc": utc_now(), "order": key}, ensure_ascii=False,
                                                         indent=1), encoding="utf-8")
    rows = []
    for k, lid, (v1, v2) in items:
        rows.append(f"""<section><h2>Line {k}</h2><audio controls preload="none" src="{_wav_b64(d / 'audio' / f'{lid}.wav')}"></audio>
<p class="v"><b>1</b> {html.escape(v1)}</p><p class="v"><b>2</b> {html.escape(v2)}</p>
<div class="c" data-line="{html.escape(lid)}">""" + "".join(
            f'<label><input type="radio" name="q{k}" value="{v}"> {t}</label>'
            for v, t in (("1", "1 is what I hear"), ("2", "2 is what I hear"), ("same", "both the same"),
                         ("cannot_tell", "cannot tell"))) + "</div></section>")
    doc = f"""<!doctype html><html lang="mr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Listening check</title><style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--line:#ddd8cc;--accent:#3d5a80}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1b1b1a;--fg:#ecebe6;--line:#3a3935;--accent:#9db4d6}}}}
body{{background:var(--bg);color:var(--fg);font:17px/1.5 system-ui,sans-serif;max-width:760px;margin:0 auto;padding:16px}}
section{{border-top:1px solid var(--line);padding:12px 0}} h2{{font-size:15px;margin:0 0 6px}} audio{{width:100%}}
.v{{margin:6px 0;font-size:19px}} .v b{{color:var(--accent);margin-right:8px}} label{{display:inline-block;margin:4px 14px 4px 0}}
button{{font:inherit;padding:8px 16px;margin:16px 0;border:1px solid var(--accent);background:none;color:var(--accent);border-radius:6px}}
</style></head><body><h1>Which version is what you hear?</h1>
<p>{len(items)} lines from your recordings. Play each line, then choose the version whose words are spelled the way the
speaker says them. The versions are in a random order; nothing tells which model wrote which. At the end press
"Save my answers" and give the downloaded file to Claude.</p>
{''.join(rows)}
<button id="save">Save my answers</button>
<script>
document.getElementById("save").onclick = () => {{
  const answers = {{}};
  document.querySelectorAll(".c").forEach(div => {{
    const r = div.querySelector("input:checked"); answers[div.dataset.line] = r ? r.value : null; }});
  const blob = new Blob([JSON.stringify({{page: "{name}", answers}}, null, 1)], {{type: "application/json"}});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "{name}.answers.json"; a.click();
}};
</script></body></html>"""
    out = COMPARE / f"{name}.html"
    out.write_text(doc, encoding="utf-8")
    return {"file": str(out), "lines": len(items), "differing_lines": len(differ)}


def score(answers_path) -> dict:
    """The owner's choices per candidate (the key file says which version was which)."""
    from pathlib import Path
    ans = json.loads(Path(answers_path).read_text(encoding="utf-8"))
    key = read_json(COMPARE / f"{ans['page']}.key.json")
    counts = {key["base"]: 0, key["candidate"]: 0, "same": 0, "cannot_tell": 0, "unanswered": 0}
    for lid, choice in ans["answers"].items():
        order = key["order"].get(lid)
        if order is None or choice is None:
            counts["unanswered"] += 1
        elif choice in ("1", "2"):
            counts[order[int(choice) - 1]] += 1
        else:
            counts[choice] += 1
    return {"base": key["base"], "candidate": key["candidate"], "counts": counts}

"""Idea engine helper.

Every judgment is made by Claude or by fresh agents: what counts as a repeat, which family each card
belongs in, which gaps are real, which ideas are worth keeping, and how new and how good each idea is.
This script only does the clerical work: gathering the collectors' files, applying those decisions,
a search box that pulls up the cards most like a given one, joining the critics' verdicts, and drawing
the pages. The search box runs a local model (mxbai-embed-large) on this Mac, so nothing is sent
anywhere; it finds candidates and never decides.

usage (from anywhere, in zsh use a function, not a string variable):
  run() { uv run -q --python 3.12 --with fastembed --with umap-learn --with scikit-learn --with numpy python ~/.claude/skills/idea-engine/scripts/engine.py "$@"; }
  run status <run>     gather raw/*.jsonl, apply my repeat decisions (repeats.json), list pairs for me to check
  run families <run>   check my family sorting (families.json + families/*.jsonl) covers every card
  run lookup <run>     start a judging round in critic/r<N>/: ideas not yet judged on their current words,
                       plus decoys and controls, under neutral codes (critic_key.json there; critics never see it)
  run judge <run>      join that round's verdicts: decoy catches, disagreements, controls vs engine, verdicts.jsonl
  run page <run>       draw map.html (one page, opens on its own)
  run picks <run>      draw picks.html: each idea in the question's terms, grouped by verdict

<run> holds question.json, raw/*.jsonl from the collectors, and later ideas.jsonl and decoys.jsonl.
Card fields: name, what, how, when, feel, example, source, domain.
"""
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter

import numpy as np

cmd, RUN = sys.argv[1], sys.argv[2]

MODEL = "mixedbread-ai/mxbai-embed-large-v1"
# How far the search box reaches when listing possible repeats for me to check, and how many of each
# card's nearest it shows. It sets what I look at, never what gets merged. On the many-projects run
# 0.82 with two per card gave about 200 pairs; real twins can score as low as 0.72, which no cutoff
# catches without thousands of pairs, so the narrow-group helpers also flag twins inside each family.
LOOK = 0.82
LOOK_TOP = 2
P = lambda *a: os.path.join(RUN, *a)  # noqa: E731
RANK = {"copy": 0, "close": 1, "fresh": 2, "outlier": 3}
VALUE = {"low": 0, "medium": 1, "high": 2}
MIN_SEARCHES = 5  # a critic line with fewer web searches than this doesn't count
broken = Counter()


def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            broken[os.path.basename(path)] += 1
    return out


def load_json(name, default):
    return json.load(open(P(name), encoding="utf-8")) if os.path.exists(P(name)) else default


def save_jsonl(name, rows):
    with open(P(name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def text_of(n):
    # What the search box compares: the mechanism and the feeling. Names and brands are left out so
    # two products doing the same thing come up together.
    return ". ".join(x for x in [n.get("what", ""), n.get("how", ""), n.get("feel", "")] if x.strip(" ."))


def card(n):
    return f"**{n['name']}** [{n['id']}] ({n.get('when', '')}): {n.get('what', '')} {n.get('how', '')}"


_model = None


def embed(texts):
    """Search-box positions for each text, remembered in the run folder so only new or changed
    descriptions get measured (the model is slow on a laptop)."""
    global _model
    cache_path = P(".embcache-" + MODEL.replace("/", "_") + ".npz")
    cache = dict(np.load(cache_path)) if os.path.exists(cache_path) else {}
    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    if todo:
        from fastembed import TextEmbedding
        if _model is None:
            _model = TextEmbedding(MODEL)
        new = np.array(list(_model.embed([texts[i] for i in todo], batch_size=16)), dtype=np.float32)
        for i, v in zip(todo, new):
            cache[keys[i]] = v
        np.savez(cache_path, **cache)
    E = np.array([cache[k] for k in keys], dtype=np.float32)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def families():
    """My family sorting: families.json holds the definitions ({"families": [{"name", "about"}]}),
    and families/*.jsonl hold one line per card ({"id", "family", "sub"}), later files winning."""
    defs = load_json("families.json", {"families": []})["families"]
    assign = {}
    files = sorted(glob.glob(P("families", "*.jsonl")), key=lambda f: (os.path.basename(f).startswith("fix"), f))
    for f in files:  # files named fix*.jsonl are read last, so my corrections always win
        for a in load_jsonl(f):
            assign.setdefault(str(a["id"]), {}).update({k: v for k, v in a.items() if k != "id"})
    # a kept card with no labels of its own inherits them from a twin merged into it
    for n in load_jsonl(P("nodes.jsonl")):
        if "family" not in assign.get(str(n["id"]), {}):
            for m in n.get("merged", []):
                if "family" in assign.get(m, {}):
                    assign.setdefault(str(n["id"]), {}).update({**assign[m], **assign.get(str(n["id"]), {})})
                    break
    return defs, assign


def idea_ids(ideas):
    """Give every idea a lasting id (i1, i2…) so verdicts never have to be matched by name."""
    used = {x["id"] for x in ideas if x.get("id")} | {x["id"] for x in load_jsonl(P("dropped.jsonl")) if x.get("id")}
    k = 1
    for x in ideas:
        if not x.get("id"):
            while f"i{k}" in used:
                k += 1
            x["id"] = f"i{k}"
            used.add(x["id"])
    return ideas


# ---------------------------------------------------------------- status
def status():
    rows = []
    for f in sorted(glob.glob(P("raw", "*.jsonl"))):
        src = os.path.splitext(os.path.basename(f))[0]
        seen = Counter()
        for n in load_jsonl(f):
            if not (n.get("how") or n.get("what")):
                continue
            # The id comes from the card's own words, so editing or reordering a raw file never
            # shifts it (runs before Oct 28, 2026 used line numbers instead).
            h = hashlib.sha1((n.get("name", "") + "|" + n.get("how", "")).encode("utf-8")).hexdigest()[:6]
            seen[h] += 1
            n.update({"found_by": src, "id": f"{src}:{h}" + (f"-{seen[h]}" if seen[h] > 1 else "")})
            rows.append(n)
    if not rows:
        print("no cards yet")
        return
    idx = {r["id"]: i for i, r in enumerate(rows)}
    dec = load_json("repeats.json", {"same": [], "different": []})
    parent = list(range(len(rows)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in dec.get("same", []):
        if a in idx and b in idx:
            ra, rb = root(idx[a]), root(idx[b])
            parent[max(ra, rb)] = min(ra, rb)
    for i, r in enumerate(rows):
        k = rows[root(i)]
        if k is r:
            continue
        k.setdefault("also_found_by", [])
        k.setdefault("merged", []).append(r["id"])  # so labels given to a twin carry over to the card kept
        if r["found_by"] != k["found_by"] and r["found_by"] not in k["also_found_by"]:
            k["also_found_by"].append(r["found_by"])
        if r.get("example") and r["example"] not in (k.get("example") or ""):
            k["example"] = ((k.get("example") or "") + "; " + r["example"]).strip("; ")
    keep = [i for i in range(len(rows)) if root(i) == i]
    save_jsonl("nodes.jsonl", [rows[i] for i in keep])

    # possible repeats: each card's few nearest above the cutoff, minus the pairs already decided
    E = embed([text_of(r) for r in rows])
    S = E @ E.T
    np.fill_diagonal(S, -1)
    different = {frozenset(p) for p in dec.get("different", [])}
    pairs = {}
    for i in range(len(rows)):
        for j in np.argsort(-S[i])[:LOOK_TOP]:
            j = int(j)
            if S[i, j] < LOOK or root(i) == root(j):
                continue
            if frozenset((rows[i]["id"], rows[j]["id"])) in different:
                continue
            pairs[frozenset((i, j))] = float(S[i, j])
    ordered = sorted(pairs.items(), key=lambda kv: -kv[1])
    with open(P("repeats_to_check.md"), "w", encoding="utf-8") as f:
        f.write("# Possible repeats to check\nFor each pair, decide: the same thing said differently, or two "
                "different things? Add the pair to repeats.json under \"same\" or \"different\".\n\n")
        for k, (pair, _) in enumerate(ordered, 1):
            i, j = sorted(pair)
            f.write(f"{k}. {card(rows[i])}\n   {card(rows[j])}\n\n")

    # completeness, in one honest number: how often separate collectors found the same thing
    kept = [rows[i] for i in keep]
    overlap = sum(1 for r in kept if r.get("also_found_by")) / len(kept)
    print(f"raw cards: {len(rows)}   after my repeat decisions: {len(kept)}")
    print(f"possible repeats still to check: {len(ordered)}  (repeats_to_check.md)")
    print(f"found by two or more collectors: {overlap:.0%}. The lower this is, the more still exists that "
          "nobody has found; near 0% means the collection is far from complete.")
    if broken:
        print("WARNING, unreadable lines skipped:", dict(broken))
    q = load_json("question.json", {})
    if not (q.get("target") and q.get("good")):
        print("WARNING: question.json has no 'target' or 'good'; finish the grilling before going further")
    weak = [r for r in kept if not r.get("how") or r.get("how", "").strip() == r.get("what", "").strip()
            or re.match(r"(?i)(this works because|because|it works)\b", r.get("how", ""))]
    if weak:
        print(f"WARNING: {len(weak)} cards have a 'how' that's empty, repeats 'what' or gives a reason instead of the mechanism, e.g.:",
              "; ".join(f"{r['id']} {r['name']}" for r in weak[:5]))
    print("by source:", dict(Counter(r["found_by"] for r in rows).most_common()))
    print("by 'when':", dict(Counter((r.get("when") or "?").split("/")[0].strip().lower() for r in kept).most_common()))
    print("by domain:", dict(Counter((r.get("domain") or "?").lower() for r in kept).most_common()))


# ---------------------------------------------------------------- families
def do_families():
    nodes = load_jsonl(P("nodes.jsonl"))
    defs, assign = families()
    ids = {str(n["id"]) for n in nodes}
    missing = [n for n in nodes if "family" not in assign.get(str(n["id"]), {})]
    unknown = [i for i in assign if i not in ids]
    with open(P("families_preview.md"), "w", encoding="utf-8") as f:
        for k, d in enumerate(defs):
            members = [n for n in nodes if assign.get(str(n["id"]), {}).get("family") == k]
            f.write(f"## {k}. {d['name']}  ({len(members)})\n{d.get('about', '')}\n")
            for sub, c in Counter(assign[str(n["id"])].get("sub", "") for n in members).most_common():
                names = [n["name"] for n in members if assign[str(n["id"])].get("sub", "") == sub]
                f.write(f"  - {sub or '(no narrow group yet)'} ({c}): " + "; ".join(names[:15]) + ("…" if c > 15 else "") + "\n")
            f.write("\n")
    print(open(P("families_preview.md"), encoding="utf-8").read()[:5000])
    print(f"{len(nodes)} cards, {len(nodes) - len(missing)} sorted, {len(missing)} not sorted yet, {len(unknown)} unknown ids")
    for n in missing[:30]:
        print("  not sorted:", n["id"], n["name"])


# ---------------------------------------------------------------- lookup
def fingerprint(x):
    """A short stamp of the words a critic judges, so a verdict never sticks to text it didn't see."""
    return hashlib.sha1((x.get("what", "") + "|" + x.get("how", "") + "|" + x.get("when", "")).encode("utf-8")).hexdigest()[:8]


def check_ideas(ideas):
    """Refuse ideas whose critic-facing words argue their own case or repeat each other."""
    problems = []
    ids = Counter(x.get("id") for x in ideas)
    for x in ideas:
        w, h = x.get("what", "").strip(), x.get("how", "").strip()
        if ids[x.get("id")] > 1:
            problems.append(f"{x['name']}: id {x.get('id')} is used twice")
        if not h or h == w or h == x.get("steal", "").strip():
            problems.append(f"{x['name']}: 'how' is empty or repeats 'what' or 'steal'")
        if re.match(r"(?i)(instead of|unlike|rather than)\b", h) or re.match(r"(?i)(instead of|unlike|rather than)\b", w):
            problems.append(f"{x['name']}: 'what' or 'how' argues against the usual way; write it plainly")
    return problems


def lookup():
    """One judging round. Ideas still to judge, the decoys and the plain-brainstorm controls are mixed
    under neutral codes, so a critic can't tell which is which, and each shows only its plain what, how
    and when (never the frame or steal written for the user). Each round gets its own folder,
    critic/r<N>/, holding the key; critics write their lines there. Every critic judges every item."""
    q = load_json("question.json", {})
    nodes = load_jsonl(P("nodes.jsonl"))
    ideas = idea_ids(load_jsonl(P("ideas.jsonl")))
    save_jsonl("ideas.jsonl", ideas)
    if not ideas:
        print("no ideas.jsonl yet")
        return
    problems = check_ideas(ideas)
    gaps = load_json("gaps_checked.json", {})
    for x in ideas:
        if gaps.get(x.get("gap")) == "fails":
            problems.append(f"{x['name']}: its gap {x['gap']} failed the gap check, so it can't be judged as an idea for it")
    if problems:
        print("Fix these before judging:\n  " + "\n  ".join(problems))
        return
    judged = {v["id"]: v for v in load_jsonl(P("verdicts.jsonl")) if v.get("id")}
    todo = [x for x in ideas if judged.get(x["id"], {}).get("fp") != fingerprint(x)]
    if not todo:
        print("every idea already has a verdict on its current words")
        return
    decoys, controls = load_jsonl(P("decoys.jsonl")), load_jsonl(P("controls.jsonl"))
    if not decoys:
        print("WARNING: no decoys.jsonl, so nothing will show whether a critic can be fooled")
    items = [("idea", x) for x in todo] + [("decoy", x) for x in decoys] + [("control", x) for x in controls]
    rnd = random.Random(len(glob.glob(P("critic", "r*"))) + 7)
    rnd.shuffle(items)
    rounds = sorted(int(os.path.basename(d)[1:]) for d in glob.glob(P("critic", "r*")) if os.path.basename(d)[1:].isdigit())
    rdir = P("critic", f"r{(rounds[-1] + 1) if rounds else 1}")
    os.makedirs(rdir)
    S = embed([text_of(x) for _, x in items]) @ embed([text_of(n) for n in nodes]).T
    key = {}
    with open(os.path.join(rdir, "critic_pack.md"), "w", encoding="utf-8") as f:
        f.write(f"# Items to judge\nThe question: {q.get('question', '')}\nThe setting: {q.get('scope', '')}\n"
                f"The real target: {q.get('target', '')}\nWhat a good answer does: {q.get('good', '')}\n\n"
                "Under each item are the 20 existing cards a search found most like it, shuffled so their order "
                "says nothing. The search can miss things, so also search all the cards in nodes.jsonl yourself.\n\n")
        for k, (kind, x) in enumerate(items):
            code = f"c{k + 1:02d}"
            key[code] = {"idea": x["id"], "fp": fingerprint(x)} if kind == "idea" else {kind: x["name"], "expect": x.get("expect", "")}
            f.write(f"## {code}: {x['name']}\nWhat: {x.get('what', '')}\nHow: {x.get('how', '')}\nWhen: {x.get('when', '')}\n\nCards like it:\n")
            near = np.argsort(-S[k])[:20]
            rnd.shuffle(near)
            for j in near:
                f.write(f"- {card(nodes[j])}\n")
            f.write("\n")
    json.dump(key, open(os.path.join(rdir, "critic_key.json"), "w"), indent=1)
    print("wrote", os.path.join(rdir, "critic_pack.md"),
          f"({len(todo)} ideas, {len(decoys)} decoys, {len(controls)} controls, codes c01 to c{len(items):02d})")
    print("critics write their lines to", os.path.join(rdir, "<critic>.jsonl"))


# ---------------------------------------------------------------- judge
def judge():
    """Join the latest round's critic lines (critic/r<N>/<critic>.jsonl, one {"code", "verdict", "value",
    "searches", ...} per item). A critic is trusted only if it judged at least one decoy and wasn't fooled
    by any. When trusted critics disagree, the less-new verdict and the lower value win, and the idea is
    marked split. Earlier verdicts on unchanged ideas are kept; the first verdict an idea ever got is
    kept beside its latest one. Controls (a plain brainstorm with no collection) get the same treatment,
    so the engine can be compared with the "poor average" it was built to replace."""
    rounds = sorted(int(os.path.basename(d)[1:]) for d in glob.glob(P("critic", "r*")) if os.path.basename(d)[1:].isdigit())
    if not rounds:
        print("no judging round yet; run lookup first")
        return
    rdir = P("critic", f"r{rounds[-1]}")
    key = json.load(open(os.path.join(rdir, "critic_key.json")))
    ideas = {x["id"]: x for x in load_jsonl(P("ideas.jsonl"))}
    by_critic = {os.path.splitext(os.path.basename(f))[0]: load_jsonl(f) for f in sorted(glob.glob(os.path.join(rdir, "*.jsonl")))}
    print(f"round r{rounds[-1]}")
    trusted = []
    for c, lines in by_critic.items():
        seen = [ln for ln in lines if "decoy" in key.get(ln.get("code"), {})]
        misses = [ln for ln in seen if RANK.get(ln.get("verdict"), 0) > RANK["copy"]]
        thin = [ln for ln in lines if len(ln.get("searches", [])) < MIN_SEARCHES]
        by_critic[c] = [ln for ln in lines if len(ln.get("searches", [])) >= MIN_SEARCHES]
        print(f"{c}: {len(lines)} of {len(key)} items, decoys seen {len(seen)}, called above copy {len(misses)}")
        if thin:
            print(f"   {len(thin)} of its lines have fewer than {MIN_SEARCHES} searches and don't count")
        for m in misses:
            print(f"   fooled by decoy {m['code']} ({key[m['code']]['decoy']})")
        if seen and not misses:
            trusted.append(c)
    use = trusted or list(by_critic)
    if not trusted:
        print("WARNING: no critic passed the decoy test; verdicts below are not reliable")

    def merge(codes):
        lines = [ln for c in use for ln in by_critic[c] if ln.get("code") in codes]
        if not lines:
            return None
        return {"verdict": min((ln.get("verdict", "copy") for ln in lines), key=lambda v: RANK.get(v, 0)),
                "value": min((ln.get("value", "low") for ln in lines), key=lambda v: VALUE.get(v, 0)),
                "split": len({ln.get("verdict") for ln in lines}) > 1,
                "nearest": sorted({n for ln in lines for n in ln.get("nearest", [])}),
                "world": [ln.get("world") for ln in lines if ln.get("world")],
                "searches": [s for ln in lines for s in ln.get("searches", [])],
                "drift": any(ln.get("drift") for ln in lines),
                "reason": " | ".join(ln.get("reason", "") for ln in lines)}

    old = {v["id"]: v for v in load_jsonl(P("verdicts.jsonl")) if v.get("id")}
    out = []
    for iid, x in ideas.items():
        codes = [c for c, v in key.items() if v.get("idea") == iid]
        m = merge(codes) if codes else None
        if m:
            first = old.get(iid, {}).get("first") or m["verdict"]
            out.append({"id": iid, "name": x["name"], "fp": key[codes[0]]["fp"], "round": rounds[-1], "first": first, **m})
        elif iid in old and old[iid].get("fp") == fingerprint(x):
            out.append(old[iid])
        else:
            out.append({"id": iid, "name": x["name"], "verdict": "not judged"})
    for c, v in key.items():
        if "control" in v:
            m = merge([c])
            ctl = next((x for x in load_jsonl(P("controls.jsonl")) if x["name"] == v["control"]), {})
            if m:
                out.append({"id": "ctl:" + v["control"], "name": v["control"], "fp": fingerprint(ctl), "round": rounds[-1],
                            "first": m["verdict"], "control": True, **m})
    save_jsonl("verdicts.jsonl", out)
    print(f"\n{len(out)} ideas: ", dict(Counter(o["verdict"] for o in out)), " split:", sum(1 for o in out if o.get("split")),
          " drift:", sum(1 for o in out if o.get("drift")))
    for o in sorted(out, key=lambda o: (RANK.get(o["verdict"], -1), o["name"])):
        changed = f" (first: {o['first']})" if o.get("first") and o["first"] != o["verdict"] else ""
        print(f"  {o['verdict']:<10} {o.get('value', ''):<7} {'SPLIT ' if o.get('split') else ''}{'DRIFT ' if o.get('drift') else ''}{o['name']}{changed}")
    controls = [(c, v) for c, v in key.items() if "control" in v]
    if controls:
        cv = [merge([c]) for c, _ in controls]
        cv = [m for m in cv if m]
        print("\ncontrols (plain brainstorm, no collection):", dict(Counter(m["verdict"] for m in cv)),
              " value:", dict(Counter(m["value"] for m in cv)))
        print("engine ideas this round:", dict(Counter(o["verdict"] for o in out if o.get("round") == rounds[-1])),
              " value:", dict(Counter(o.get("value") for o in out if o.get("round") == rounds[-1])))


def verdicts_by_id():
    """Verdicts by idea id, dropping any given on words the idea no longer has."""
    ideas = {x.get("id"): x for x in load_jsonl(P("ideas.jsonl"))}
    return {v["id"]: v for v in load_jsonl(P("verdicts.jsonl"))
            if v.get("id") and (not v.get("fp") or v["fp"] == fingerprint(ideas.get(v["id"], {})))}


# ---------------------------------------------------------------- page
def page():
    import umap
    q = load_json("question.json", {})
    nodes = load_jsonl(P("nodes.jsonl"))
    defs, assign = families()
    ideas = load_jsonl(P("ideas.jsonl"))
    vid = verdicts_by_id()
    vname = {v["name"]: v for v in load_jsonl(P("verdicts.jsonl")) if not v.get("id")}  # runs before Oct 28 matched by name
    for x in ideas:
        v = vid.get(x.get("id")) or vname.get(x["name"], {})
        x.update({"verdict": v.get("verdict", "not judged"), "nearest": v.get("nearest", []), "reason": v.get("reason", "")})
    # the dots' positions are only a picture: the search box places alike cards near each other
    E = embed([text_of(n) for n in nodes] + [text_of(x) for x in ideas])
    xy = umap.UMAP(n_neighbors=12, min_dist=0.25, metric="cosine", random_state=7).fit_transform(E)
    xy = (xy - xy.min(axis=0)) / (xy.max(axis=0) - xy.min(axis=0))
    fam = [assign.get(str(n["id"]), {}) for n in nodes]
    pts = [{"x": float(xy[i, 0]), "y": float(xy[i, 1]), "g": fam[i].get("family", -1), "sub": fam[i].get("sub", ""),
            "n": n["name"], "w": n.get("what", ""), "h": n.get("how", ""), "f": n.get("feel", ""),
            "e": n.get("example", ""), "s": n.get("source", "")} for i, n in enumerate(nodes)]
    ids = [{"x": float(xy[len(nodes) + k, 0]), "y": float(xy[len(nodes) + k, 1]), **x} for k, x in enumerate(ideas)]
    gl = []
    for g, d in enumerate(defs):
        m = [i for i in range(len(nodes)) if pts[i]["g"] == g]
        if m:
            gl.append({"id": g, "name": d["name"], "count": len(m),
                       "x": float(np.median(xy[m, 0])), "y": float(np.median(xy[m, 1]))})
    data = {"q": q, "points": pts, "ideas": ids, "groups": gl}
    tpl = open(os.path.join(os.path.dirname(__file__), "page.html"), encoding="utf-8").read()
    open(P("map.html"), "w", encoding="utf-8").write(tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    print("wrote", P("map.html"))


# ---------------------------------------------------------------- picks
def picks():
    """The hand-over page: every idea grouped by verdict, with no closest match beside it. Copies that
    are still good for the question go under 'Just use this' instead of disappearing."""
    q = load_json("question.json", {})
    vid = verdicts_by_id()
    vname = {v["name"]: v for v in load_jsonl(P("verdicts.jsonl")) if not v.get("id")}  # runs before Oct 28
    ideas = []
    for x in load_jsonl(P("ideas.jsonl")):
        v = vid.get(x.get("id")) or vname.get(x["name"], {})
        verdict = v.get("verdict", "not judged")
        if verdict == "copy":
            verdict = "use" if VALUE.get(v.get("value"), 0) >= VALUE["medium"] else "copy"
        ideas.append({**{k: x.get(k, "") for k in ["id", "name", "frame", "steal", "ways", "mock", "sample", "try"]},
                      "verdict": verdict})
    # proven ideas from the blind control brainstorm, judged medium or high value, go in labelled
    for v in load_jsonl(P("verdicts.jsonl")):
        if v.get("control") and VALUE.get(v.get("value"), 0) >= VALUE["medium"]:
            ctl = next((x for x in load_jsonl(P("controls.jsonl")) if x["name"] == v["name"]), {})
            ideas.append({"id": v["id"], "name": v["name"], "frame": "From the plain brainstorm, judged blind",
                          "steal": ctl.get("what", ""), "ways": [ctl.get("how", "")], "mock": ctl.get("mock", ""),
                          "verdict": "use" if v["verdict"] == "copy" else v["verdict"]})
    tpl = open(os.path.join(os.path.dirname(__file__), "picks.html"), encoding="utf-8").read()
    data = {"q": q, "ideas": ideas, "has_map": os.path.exists(P("map.html"))}
    open(P("picks.html"), "w", encoding="utf-8").write(tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    print("wrote", P("picks.html"))


{"status": status, "families": do_families, "lookup": lookup, "judge": judge, "page": page, "picks": picks}[cmd]()

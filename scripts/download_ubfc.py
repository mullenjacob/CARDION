"""Download UBFC DATASET_2 subjects from the Drive index (resumable, skips existing)."""
import argparse, json, os, sys, time, gdown
ap = argparse.ArgumentParser()
ap.add_argument("--index", default="data/raw/ubfc_drive_index.json")
ap.add_argument("--out", default="data/raw/ubfc")
ap.add_argument("--subjects", nargs="*", default=None, help="e.g. subject1 subject3; default all")
a = ap.parse_args()
idx = [e for e in json.load(open(a.index)) if e["path"].startswith("DATASET_2/")]
for e in idx:
    subj = e["path"].split("/")[1]
    if a.subjects and subj not in a.subjects:
        continue
    dest = os.path.join(a.out, e["path"])
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("skip", e["path"]); continue
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    t = time.time()
    r = gdown.download(id=e["id"], output=dest, quiet=True, use_cookies=False)
    if r is None or not os.path.exists(dest):
        print("FAILED", e["path"]); sys.exit(2)
    mb = os.path.getsize(dest) / 1e6
    print(f"ok {e['path']} {mb:.1f} MB in {time.time()-t:.1f}s ({mb/(time.time()-t):.1f} MB/s)", flush=True)

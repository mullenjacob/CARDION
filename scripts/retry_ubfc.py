"""Background retry loop for UBFC videos: tries every 30 min for up to 24 h, stops when all videos exist."""
import subprocess, sys, time, glob
for i in range(48):
    subprocess.run([sys.executable, "scripts/download_ubfc.py"], capture_output=True)
    n = len(glob.glob("data/raw/ubfc/DATASET_2/*/vid.avi"))
    print(time.strftime("%H:%M:%S"), "videos present:", n, flush=True)
    if n >= 42: break
    time.sleep(1800)

# CARDION user guide

CARDION estimates your heart rate from a camera by looking for the tiny colour change the pulse causes in your skin, and compares
several algorithms on the same input, including a spiking neural network that works on *simulated* event-camera data.
It is a **research prototype, not a medical device**. No event camera is involved: the "events" are computed from your normal camera video.

## Start it (Windows)

1. Double-click **`Launch CARDION.bat`**.
2. Every launch first runs a quick health check (a few seconds). If something is missing or broken (an interrupted download, a damaged package,
   a damaged Python environment) CARDION **repairs it by itself**. The very first time it downloads Python and the needed packages
   (about 1.5 GB, 5-15 minutes, internet required **once**). Everything is installed inside this folder (`.venv`, `tools`); nothing else on
   your PC is changed. The folder may have spaces in its path and may live anywhere you can write (not `Program Files`).
3. Your browser opens at `http://127.0.0.1:8765/` (the page only works on your own computer). Keep the black console window open;
   close it to quit CARDION.

**If something goes wrong** you do not need a terminal: the window says what failed and where the complete error report was saved
(the `logs` folder next to the `.bat` files; the newest report is opened in Notepad for you). Nothing is cut off.

Other double-click files:

* **`Test and Repair CARDION.bat`** - diagnoses and repairs the installation, runs the self-test, tests the camera pipeline with a recorded
  face (no hardware), then tests your real webcam (the camera light turns on for ~20 s; sit in front of it in good light). One report:
  `out/test_and_repair_report.md`. A missing or blocked webcam is a warning with advice, not a failure.
* **`Run CARDION Self-Test.bat`** (about 3 minutes) - simulated face, test video, recording/export, error handling, camera code path with a
  recorded face; PASS/FAIL report (also `out/selftest_report.md`).

## Try it without a camera

Open the **Built-in demo** tab and press **Start**. A simulated face with a *known* pulse is analysed; the page shows the true rate so
you can see the estimate converge to it after about 8-10 seconds. If no camera is found at all, CARDION selects the demo for you.

## Use your webcam

1. **Live camera** tab -> pick your camera (found automatically) -> **Start**. Windows may ask for camera permission the first time
   (Settings > Privacy & security > Camera > allow desktop apps). CARDION never records unless you press *Start recording*.
2. Face the camera, 0.5-1 m away, with steady, bright light on your face (a window in front of you is ideal). Keep still and do not talk.
3. After about 8 seconds the number appears, but only if the quality checks pass. Otherwise it tells you why (too dark, too much
   movement, pulse too weak, methods disagree...). **A missing number is intentional**: CARDION does not guess.

### Guided positioning (live camera)
When you press **Start**, CARDION watches where your face is and tells you, in short sentences, how to move: "Move slightly right.", "Move a little closer.", "Face the camera directly.", "Hold still for a moment.", "Try more even lighting." and so on. Directions are from **your** point of view and the live picture is shown like a mirror, so an arrow pointing right means "move right".
- A dashed box marks your face; a faint dashed target shows roughly where it should be. When the framing is fine the box turns solid and the message becomes "Great positioning. Measuring your pulse."
- CARDION does not know how far away you are (webcams do not measure distance). It only checks how much of the picture your face fills.
- Advice follows the signal, not looks: if your face is a bit off-centre but the pulse is being read reliably, it will not nag you.
- Instructions only change after a condition has lasted about a second, so they do not flicker.
- If more than one person is in view, no heart rate is shown. The solid blue box is the face being measured; dashed red boxes are other faces, which are ignored. CARDION never identifies anyone or guesses age, sex or health from a face.
- The progress strip under the picture and the four steps (Position - Collect - Stabilize - Reliable) show where you are. "Measurement quality" lists framing, face tracking, lighting, movement, pulse signal, agreement between methods and amount of data separately: good framing alone does not make a reading accurate.
- If the camera itself delivers few frames per second (common in a dim room), CARDION says so; more light usually fixes it.

## Analyse a video

**Video file** tab -> *Choose video...* (AVI/MP4/MOV/MKV) -> pick *Real time*, *4x* or *As fast as possible* -> **Start**.
A face with good, steady lighting works best; compressed/dark videos often hide the pulse.

## What you see

* **Big number** - the average of the methods that agree, with their spread. **Confidence** bar and a status badge.
* **Signal quality** - data collected, face tracked, movement, brightness, light stability, number of agreeing methods.
* **Pulse waveforms / history** - the extracted pulse of each method (last 10 s) and heart-rate over time (white = reliable estimate).
* **Methods table** - RGB green, RGB POS, RGB CHROM, event green, event POS (classical), the spiking network and the matched ANN, each
  with its own BPM, signal strength and compute cost (milliseconds per second of video and operation counts).
  Energy figures are *operation-count proxies*, not measured power.
* **Event view** - red/blue dots where the simulated sensor sees brightness go up/down.

## Record and export

Press **Start recording** (optionally tick *save annotated video*), later **Stop recording**. Each session is saved under
`sessions/session_<time>/` with `estimates.csv` (every second: BPM of each method, status, quality), `summary.json`, `plots.png`
(and `annotated.mp4`). The list on the page offers a `.zip` download; **Open sessions folder** opens it in Explorer.

## Troubleshooting

| Symptom | What to do |
|---|---|
| Setup stops with a download error | Check the internet connection and run the launcher again; it resumes. Technical details: newest `logs\launcher_*.log` / `logs\doctor_*.log`. |
| Any other start-up failure | Double-click `Test and Repair CARDION.bat`; read `out\test_and_repair_report.md`. Send the newest files in `logs\` if you ask someone for help. |
| Folder is very deep / path > 100 characters | Move the CARDION folder to something short such as `C:\CARDION` (Windows long paths are off on many PCs). |
| "Camera could not be opened" | Close other apps using the camera; allow camera access for desktop apps in Windows settings; press *Rescan*. |
| Number never appears | Read the badge under the number; improve light, sit still, retry. Dark skin tones and dim light reduce the pulse signal for every camera-based method. |
| Browser did not open | Open the address printed in the console window. |
| Port busy | CARDION tries the next ports (8766, ...); use the address it prints. |
| Windows blocks the downloaded setup tool (`uv.exe`) | The launcher then uses a Python 3.10-3.12 already installed on the PC (python.org) to build `.venv` and installs with pip. If none is installed, install Python 3.11 from python.org once and launch again. |
| Computer too slow / video stutters | Untick **event & neural methods** (the heaviest part); CARDION also pauses them automatically on a live camera if it falls behind. |
| Want to reinstall | Delete the `.venv` folder and launch again (your models, data and results are not in it). |
| Camera light flashes when CARDION starts | Camera discovery briefly opens each camera. It never records unless you press *Start recording*. |

## Privacy

Video is processed on your computer only; nothing is uploaded. Uploaded videos are copied to `sessions/uploads/`. Delete the `sessions`
folder to remove saved data.


## Validation mode: compare CARDION with your Apple Watch (or another monitor)

Open **Launch CARDION.bat**, then press **Validation mode** (top right). No spreadsheets or commands are needed.

1. **Set up:** the anonymous code is filled in (P001, P002...). Choose the reference device (Apple Watch, ECG chest strap...), the condition and the length (60 s is recommended). Tick the consent box only after reading `docs/VALIDATION_PROTOCOL.md` section 5 if anyone other than yourself is being measured. Press **Start timed measurement**.
2. **Measure:** the camera, positioning help and the six start-up steps appear as usual. At the prompts (20 s, 35 s, 50 s) read your watch, type the number, press **Enter**. If you looked a few seconds earlier choose "about N s ago". Typing a wrong number? **Edit** or **Delete** it - every change is logged.
3. **Result:** when the timer ends the report appears: mean absolute error, bias, how often CARDION was within 1/3/5 BPM, **and how many of your watch readings could be compared at all**, the time to the first reliable reading, withheld periods and why. **Open full report** gives a printable page; **Download session (.zip)** has CSV and JSON files.

Things to know:

* A watch reading CARDION could not match (no reliable CARDION number within +/-5 s) is counted as "not available", never silently dropped or treated as an error.
* The accuracy is only as good as the reference. A wrist watch is not an ECG.
* Several sessions from one person are **not** several independent people; the Study overview counts each person once and shows an interval only with 8+ participants.
* The "Built-in simulated face" source is for testing the software; its results are excluded from the study summary.
* You can add readings later (from the Health app) by typing a time in seconds, or import a CSV (`time_s,bpm` or `timestamp,bpm`) / Apple Health `export.xml`.
* Files live in `validation_data/<participant>/<session>/` (private; not uploaded anywhere; video is saved only if you tick the box).

## Smoother live view (what changed in session 7)

* The BPM readout now says what it is: **Current reliable estimate**; **Latest confirmed estimate** (a second or two old, re-checking); **Older estimate ... not current** (dimmed, struck through, with its age and the reason); **Not enough data yet**; or **Signal rejected: reason**. No number is invented or smoothed.
* The start-up checklist (Camera connected, Face detected, Positioning suitable, Collecting pulse data, Stabilizing, Reliable BPM available) is driven by measured state. The progress bar counts seconds of *usable face data*, not a timer.
* The first reliable number needs at least 8 s of face data plus agreement between methods; this wait is deliberate and was not shortened.

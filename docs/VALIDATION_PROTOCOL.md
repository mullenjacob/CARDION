# CARDION validation protocol (pilot study)

This is the plan for finding out whether CARDION's very close agreement with a watch in informal tests (0-1 BPM) is **repeatable across people and
conditions**. It is written so a student researcher can run it with the **Validation mode** page (`Launch CARDION.bat` -> "Validation mode"),
and so it can be extended to more people later without changing anything.

> **Read this before recruiting anyone** - see "Consent, privacy and research ethics" below. Do not start collecting other people's data until the
> required approval is in place.

## 1. Question and what counts as an answer

*Primary question:* for resting adults seated in front of a normal webcam, how far is the heart rate CARDION displays from an independent
reference monitor, **and how often does CARDION display a reliable number at all?**

Both halves are always reported together. CARDION withholds a number when the signal is poor (by design); a method that only answers when it is
easy can look very accurate while being rarely available. Reported per participant and overall:

| outcome | meaning |
|---|---|
| MAE, RMSE, signed bias, median / max absolute error | agreement on *paired* readings only |
| share within 1 / 3 / 5 BPM | how often the error is small |
| **paired share** | fraction of reference readings that had a reliable CARDION reading within +/- 5 s |
| **reliable share** | fraction of session time (after the 8 s warm-up) CARDION showed a reliable number |
| time to first reliable reading, withheld periods and reasons | the waiting experience |
| steadiness | how much the shown number changes second to second |

Success is **not** "MAE below X". The pilot is descriptive. Agreement is judged against what the reference device itself can deliver (see 6).

## 2. Design and why

| choice | value | reason |
|---|---|---|
| Participants | about **10** to start; keep adding | A pilot of 10 can show whether the informal result is plausible and expose failure cases (skin tone, glasses, beards...). It cannot give a tight population estimate (see table below). The page is built so more can be added any time. |
| Sessions per participant | **3 resting sessions** | Repeated sessions show within-person repeatability. Repeated readings from one person are *not* independent subjects; analysis counts each person once. |
| Session length | **60 s** (90-180 s available) | CARDION needs 8 s of data before its first estimate and uses 10 s windows, so a first reading comes at about 9-12 s at best. 60 s leaves about 50 one-per-second estimates and room for three reference readings. Longer sessions give more pairs but tire the participant. |
| Reference readings | **3 per session** (screen prompts at 20, 35, 50 s) | Early enough that CARDION can answer, spaced so readings are not duplicates. |
| Resting before | **3 min seated** | Heart rate settles; the reference changes little during the session, so small timing errors in hand-entered values matter less. |
| Pairing window | **+/- 5 s** around the time of the reference reading; the report also shows 3 s and 10 s | A watch value is averaged over a few seconds and typed a few seconds later; CARDION's window is 10 s long. The sensitivity table shows whether conclusions depend on this choice. |
| Reference | **ECG chest strap preferred** (commonly used as a reference in wearable validation studies); Apple Watch acceptable | A watch is an optical wrist sensor with its own error, not clinical ECG. Record which device and wrist. If the watch and CARDION disagree, the watch may be the one that is off. |
| Conditions (optional extras) | small head movements; dim light; bright light | One condition per session, labelled in the page. Run the resting sessions first. |
| Another day | enter a different "Visit / day" label | Tests repeatability over time. |
| Source | **Live camera** only for real results | The built-in simulated face and video files are for software testing and are excluded from the study summary. |

### How precisely can N participants pin down the average error?

Planning value from the existing validation recordings (`scripts/protocol_precision.py`, output in `out/protocol_precision_*.json`): the
between-person standard deviation of per-person MAE of the gated number was **10.5 BPM** across 12 research subjects (some people measure much worse
than others). With that spread the 95 % interval for the *mean* of per-participant MAE is about:

| participants | +/- BPM |
|---|---|
| 5 | 13.0 |
| 10 | 7.5 |
| 20 | 4.9 |
| 30 | 3.9 |

This is a planning assumption from a different dataset, not a prediction for your setup. If your people are all measured well, the spread (and
the interval) will be far smaller - but that is something the pilot has to *show*. With 10 people, report the **number of participants whose
MAE was within 3 BPM** (for example 9 of 10) rather than a precise mean, and report the per-person table. The app refuses to draw a
confidence interval with fewer than 8 participants, and refuses Bland-Altman limits of agreement with fewer than 20 paired readings.

## 3. Run-sheet for each session

1. Rest 3 minutes. Seat the participant at arm's length, face evenly lit (window or lamp in front, not behind), looking at the camera. Put the
   reference monitor where you can read it without the participant moving.
2. Open **Validation mode**. Use the next anonymous code (P001, P002...). Pick the device, condition, visit, 60 s. Tick the consent box only if
   the steps in section 5 are complete. Leave "save preview video" unticked.
3. Press **Start timed measurement**. Wait; the screen shows camera, face, framing and data-collection progress.
4. At each prompt read the monitor, type the number, press **Enter**. If you read it a few seconds earlier choose "about N s ago".
5. When the timer ends the report opens by itself. Check the headline, then **Download session (.zip)** or open the data folder
   (`validation_data/<participant>/<session>/`).
6. If a number was typed wrongly, **Edit** it - the change is logged and the original is kept. Entries can also be added afterwards (for example from
   the Health app history) with a time in seconds, or imported from a CSV / Apple Health `export.xml`.
7. After all sessions press **Refresh overview** on the same page for the participant-level summary (`validation_data/exports/`).

## 4. Analysis plan (fixed before collecting data)

* Pair each reference entry with the nearest *reliable* CARDION estimate within +/- 5 s. Entries with none are **unpaired**: excluded from error
  statistics, counted in the paired share with the reason (collecting, weak signal, methods disagree, movement, light...). Nothing is interpolated.
* Per session: metrics above plus the sensitivity to the pairing window and, for context, what each individual method (no reliability gate) would
  have said at the same moments.
* Per participant: pool that person's pairs; the study headline is the **mean (and median) of per-participant MAE**, each person counted once. A
  bootstrap interval over participants appears only with >= 8 participants.
* Report availability next to every accuracy number. Report failed or interrupted sessions; do not delete them.
* No algorithm, threshold or model is changed using this data. If changes are wanted later, use a new, separate set of participants for the check.

## 5. Consent, privacy and research ethics (check **before** recruiting)

This is guidance, not legal advice. Rules differ by school, competition, country and age of participants - **ask your teacher / research
mentor first.**

* **Science fairs and competitions.** Fairs affiliated with Regeneron ISEF require that student research with human participants is reviewed and
  approved by an Institutional Review Board (IRB) *before any interaction, including recruitment and data collection*; a school IRB
  can usually do this for projects run at school, home or in the community. Check the current rulebook of your fair
  ([Society for Science ISEF rules FAQ](https://www.societyforscience.org/isef/international-rules/faq/)). Data collected before approval may be
  ineligible.
* **Minors.** If any participant is under 18 you typically need written parental/guardian permission *and* the minor's own agreement (assent). The
  IRB decides the exact form. Do not rely on a friend saying "sure".
* **Informed consent** should say, in plain words: what is measured (a webcam and a heart-rate reading), how long it takes, that CARDION is a
  research prototype and **not a medical device** (no health advice or diagnosis is given), what is stored (anonymous code, heart-rate numbers,
  timestamps; no video unless separately agreed), who sees it, that taking part is voluntary and can stop at any time, and how to withdraw data.
* **Do not give health advice.** If a reading looks unusual, do not interpret it; suggest the person speak to a doctor if they have concerns.
* **Privacy.** Validation mode stores only an anonymous code (P001) with the numbers. Keep the key linking codes to people (if you need one) on
  paper, separate from the data, and destroy it when no longer needed. No face recognition is used; the "preview video" option is off by default
  and shows the participant's face - use it only if the IRB approves and the participant agrees. Keep `validation_data/` private (it is excluded
  from git). Note that MediaPipe's native library has been seen attempting a network log call at start-up (see NOTES.md); CARDION itself sends no
  video or readings anywhere. If that matters to your IRB, run on an offline machine.
* **Safety.** Seated, resting measurements only unless the IRB approves more. The "after light exercise" condition needs explicit approval.
* **Publication / competition.** State plainly that the study is a small pilot, the events are simulated, the reference is a consumer device (if
  so), and the app is not a medical device. Do not describe it as validated for clinical use.

## 6. What the pilot cannot tell you

* Agreement with ECG if you only use a watch; the watch error is part of every difference.
* Performance for darker skin tones, glasses, facial hair, moving subjects or poor light **unless you deliberately include them** (record them in
  the condition / device note and report them separately).
* Anything about real event-camera hardware: the events are simulated from webcam video, and the displayed number does not use them at all (it is
  the average of the agreeing RGB methods).
* Generalisation from a handful of friends to the public.

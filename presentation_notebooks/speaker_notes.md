# Speaker notes — 5-minute presentation

**Total budget: 5:00** across 7 main slides (+1 hidden backup). Practice with a stopwatch 3 times before the real thing. The first run will overshoot — aim for 4:50 in rehearsal so the real run lands on 5:00.

Every "**[cut]**" marker is a sentence you can drop if you are running long.

---

## Slide 1 — Title (target 0:10)

> "Hi, I'm [name]. Today I'm going to show you how we used deep learning to automatically detect primate vocalizations in multi-hour field recordings from Makokou, Gabon."

*(Click to next slide.)*

---

## Slide 2 — The problem (target 0:40, cumulative 0:50)

> "Bioacoustic monitoring — leaving microphones in the forest and listening to them later — is one of the most powerful tools we have for non-invasive wildlife monitoring. A single recorder can run for weeks and capture animals we'd never see with the human eye.

> The problem is scale. A week of continuous recording produces **hundreds of hours** of audio, and a human analyst can listen to maybe one or two hours a day. **[cut]** So most of that audio is never actually analyzed.

> Our goal is to automate one piece of this: given a long field recording, find every time one of three forest primates — *Cercopithecus nictitans*, *Colobus guereza*, and *Pan troglodytes* — vocalizes, and classify which species it is."

*(Click.)*

---

## Slide 3 — Training data + augmentation (target 1:10, cumulative 2:00)

> "Our training set is about **870 species clips** and **1,800 background clips**. The left panel shows raw versus augmented counts per species; the right panel shows what the augmentation actually does to a single clip.

> A short note on preprocessing, because it's important for what the model is actually learning. The alarm calls we target are very short — *pyow* is around 0.1 second, *hack* is around 0.07 second, *kek* is around 0.04 second (that's Mehon & Stephan, *Royal Society Open Science* 2021). Rather than zero-pad them to the 5-second input length, we do one of two things: either place each call at a random position inside a 5-second window padded with **real ambient noise** — different noise each time, so the model can't learn a silence tell — or **concatenate 2–3 calls** of the same type with 1.5–3 second gaps, which also mirrors the natural alarm-call sequences described in that paper.

> **[cut if tight]** On top of that, augmentation gives us a ×7 multiplier per clip: original, three background-noise mixes at randomised SNR between −5 and +10 dB, one time-axis crop, one frequency-axis crop, and one ±20 mel-bin frequency shift. That brings us to roughly **6,000 effective training samples**, and teaches the model to be invariant to background, timing, and small pitch shifts."

*(Click.)*

---

## Slide 4 — The method (target 0:50, cumulative 2:50)

> "The core idea is simple: treat audio classification as **image classification**. We take a 5-second sliding window over the audio, convert it to a mel-spectrogram, resize it to 224 by 224, and feed it to a VGG19 network that was pre-trained on ImageNet.

> This works because the low-level features VGG19 learned on natural images — edges, textures, repeating patterns — are exactly the features that distinguish spectrograms of different vocalizations. **[cut]** We freeze the ImageNet backbone and only train a small custom head on top: global average pooling, a 512-unit dense layer, a 256-unit dense layer, and a 4-way softmax."

*(Click.)*

---

## Slide 5 — Model results (target 1:00, cumulative 3:50)

> "On a held-out validation set, the model reaches **94.3% accuracy** across the four classes — that's the big number on the left.

> The confusion matrix on the right tells a more nuanced story. Look at the diagonal: *Cercopithecus* and *Colobus* recall is very high, above 90%. The hardest class is actually the background class, where a small fraction of samples get mis-labelled as *Cercopithecus* — which makes sense, because many of those background clips contain distantly-calling monkeys that sound similar.

> **[cut]** Per-class F1 scores are all above 0.9 for the species; the biggest weakness is background precision."

*(Click.)*

---

## Slide 6 — Field deployment (target 0:50, cumulative 4:40)

> "But validation accuracy on clean 5-second clips is not the metric that matters. What matters is: does it work on a real, messy, multi-hour field recording?

> To test this we ran the trained model on 13 recordings from Makokou from a single day, June 9th 2022, spanning morning to night. That's about **[X] hours** of audio, around 10,000 sliding-window predictions.

> One practical surprise: the default confidence threshold of 0.7 gave us almost nothing. A threshold sweep showed that the model's confidence is bimodal — it's either very sure or it assigns the window to background. We settled on 0.4, below which the detection count saturates.

> At 0.4 the model returned **44 detections** across the 13 files. Parsing the timestamp out of each filename lets us look at when in the day each species was active — you can see here that *Pan troglodytes* peaks in [hour] and *Colobus* in [hour]. **[cut]** These diurnal patterns match published field observations.

> *(If audio is available:)* And here's one of those detected clips — this is exactly what the model flagged:" *(play 2-3 seconds of a clip from `outputs/detected_clips/Pan_troglodytes/`)*.

*(Click.)*

---

## Slide 7 — Takeaways + next steps (target 0:20, cumulative 5:00)

> "Three things to take away:

> 1. VGG19 transfer learning works well for primate vocal classification — 94% on clean clips.
> 2. Deployed end-to-end on real 2022 field recordings, it recovers meaningful diurnal activity patterns.
> 3. The whole pipeline is released as a **reusable Python package** — any researcher with their own primate recordings can run it by editing one config file.

> Main open challenge: *Pan troglodytes* recall is lower than we'd like. Next steps are more field data, per-call-type classification, and temporal context across adjacent windows.

> Thank you — happy to take questions."

**Note on the package bullet:** the value framing ("any researcher can run it on their own data") is what sticks. Do **not** list module names aloud — that's what the backup slide is for. If a Q&A question pushes for detail, pull up `figures/package_structure.png` as your backup slide and talk through `config.py` + one or two others.

---

## Backup slide — package structure (not in main flow)

There is an extra slide at the end of the deck using `figures/package_structure.png`. **Do not include it in the 7-slide flow** — keep it hidden at the end. Pull it up only if someone asks a code / reproducibility question.

If you do use it, the 30-second script is:

> "The whole pipeline lives in `src/` as 8 Python modules — one per pipeline stage. `config.py` is the entry point: all paths, hyper-parameters, and the list of species folders live there. To run this on a new dataset, the only file you edit is `config.py`. Everything else — loading, augmentation, training, detection — is dataset-agnostic."

Do not read the table row by row — let the audience read it while you summarise.

---

## Backup answers for likely questions

**Q: Why VGG19 and not a newer architecture like EfficientNet or ViT?**
> "VGG19 is a conservative choice — we wanted something well-understood, with stable ImageNet weights, and enough capacity for the task without being overkill. For 4 classes on ~6,000 augmented samples, we didn't think the extra capacity of a modern architecture would help and it would have made training slower. Moving to a newer backbone is a reasonable next step once we have more data."

**Q: How do you know the detections are real and not false positives?**
> "We manually listened to every detection in this run. *(Then quote your real count of true positives.)* The confidence threshold and the class balance in the threshold sweep are both evidence that the model isn't randomly triggering, but manual validation is still required."

**Q: Why include an 'other primate' class in background instead of a separate class?**
> "Three reasons: we don't have enough labelled samples of *Cercocebus torquatus* to train it as a first-class species, treating it as background teaches the model to actively suppress it, and the downstream task is 'find the three target species' so anything else is by definition background."

**Q: Does it work at night / in rain?**
> "The training data is mostly daytime recordings in dry conditions, so domain shift to night or heavy rain is an open concern. The Makokou recordings we tested on span 8am to 11:30pm and the model still returned species-level detections through that range, which is encouraging but not conclusive."

**Q: How long does it take to run on a full day of audio?**
> "On a T4 GPU, a 30-minute recording runs through the sliding window inference in roughly a minute. So a 24-hour recording is something like 40-60 minutes of GPU time — easily overnight-able."

---

## Rehearsal checklist

- [ ] First run, time yourself. You will be over 5:30. Don't panic.
- [ ] Second run, cut Slide 3 by 20 seconds (drop the augmentation paragraph — the figure on the right speaks for itself) and Slide 6 by 10 seconds. Aim for 5:10.
- [ ] Third run, on your feet, aim for 4:50.
- [ ] Verify the demo audio clip works on the presentation laptop **before** the talk.
- [ ] Have the PNG files downloaded locally — don't rely on live Colab during the talk.
- [ ] Know which 3 sentences you'll cut if Slide 6 runs long (the backup is: skip the threshold sweep story entirely and jump straight to the diurnal pattern figure).
- [ ] Slide 3 is the most content-dense slide in the deck — if you're short on time, the preprocessing story (ambient noise padding + concatenation) can become one sentence: "short calls get embedded in real ambient noise at a random position, or concatenated into natural sequences".

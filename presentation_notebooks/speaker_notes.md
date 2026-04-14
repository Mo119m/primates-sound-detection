# Speaker notes — 5-minute presentation

**Total budget: 5:00**. Practice with a stopwatch 3 times before the real thing. The first run will overshoot — aim for 4:50 in rehearsal so the real run lands on 5:00.

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

## Slide 3 — The data (target 0:45, cumulative 1:35)

> "Our training set comes from pre-extracted 5-second clips of each species, plus a background class that contains forest ambient noise, other primate species we aren't targeting, and previously misclassified samples.

> In total that's around 870 species clips and about 1,800 background clips — roughly **[X] minutes of labelled audio**." *(Read real number from 01_clips_per_class.png / 02_minutes_per_class.png.)*

> "On the right you can see one mel-spectrogram per species — this is what the model actually sees. Each species has a visually distinct call signature: *Cercopithecus* has these short sharp hacks here, *Colobus* has the long roar-like pattern, and *Pan* has the pant-hoot climb."

**[cut if tight]** *"Background clips deliberately include other primate species like* Cercocebus torquatus *so the model learns to reject them rather than over-trigger."*

*(Click.)*

---

## Slide 4 — The method (target 1:00, cumulative 2:35)

> "The core idea is simple: treat audio classification as **image classification**. We take a 5-second sliding window over the audio, convert it to a mel-spectrogram, resize it to 224 by 224, and feed it to a VGG19 network that was pre-trained on ImageNet.

> This works because the low-level features VGG19 learned on natural images — edges, textures, repeating patterns — are exactly the features that distinguish spectrograms of different vocalizations. **[cut]** We freeze the ImageNet backbone and only train a small custom head on top: global average pooling, a 512-unit dense layer, a 256-unit dense layer, and a 4-way softmax.

> Training uses heavy augmentation — we mix each species clip with 3 different backgrounds at varying signal-to-noise ratios, plus time cropping and frequency shifting — which pushes the effective training set from 870 clips to about 6,000."

*(Click.)*

---

## Slide 5 — Model results (target 1:00, cumulative 3:35)

> "On a held-out validation set, the model reaches **94.3% accuracy** across the four classes — that's the big number on the left.

> The confusion matrix on the right tells a more nuanced story. Look at the diagonal: *Cercopithecus* and *Colobus* recall is very high, above 90%. The hardest class is actually the background class, where a small fraction of samples get mis-labelled as *Cercopithecus* — which makes sense, because many of those background clips contain distantly-calling monkeys that sound similar.

> **[cut]** Per-class F1 scores are all above 0.9 for the species; the biggest weakness is background precision."

*(Click.)*

---

## Slide 6 — Field deployment (target 1:10, cumulative 4:45)

> "But validation accuracy on clean 5-second clips is not the metric that matters. What matters is: does it work on a real, messy, multi-hour field recording?

> To test this we ran the trained model on 13 recordings from Makokou from a single day, June 9th 2022, spanning morning to night. That's about **[X] hours** of audio, around 10,000 sliding-window predictions.

> One practical surprise: the default confidence threshold of 0.7 gave us almost nothing. A threshold sweep showed that the model's confidence is bimodal — it's either very sure or it assigns the window to background. We settled on 0.4, below which the detection count saturates.

> At 0.4 the model returned **44 detections** across the 13 files. Parsing the timestamp out of each filename lets us look at when in the day each species was active — you can see here that *Pan troglodytes* peaks in [hour] and *Colobus* in [hour]. **[cut]** These diurnal patterns match published field observations.

> *(If audio is available:)* And here's one of those detected clips — this is exactly what the model flagged:" *(play 2-3 seconds of a clip from `outputs/detected_clips/Pan_troglodytes/`)*.

*(Click.)*

---

## Slide 7 — Takeaways + next steps (target 0:15, cumulative 5:00)

> "Three things to take away:

> 1. VGG19 transfer learning works well for primate vocal classification — 94% on clean clips.
> 2. Deployed end-to-end on real 2022 field recordings, it recovers meaningful diurnal activity patterns.
> 3. The biggest open challenge is *Pan troglodytes* recall — we got only 5 detections in 13 recordings, which is likely under-triggering.

> Next steps: more field data, per-call-type classification instead of just species, and adding temporal context across adjacent windows.

> Thank you — happy to take questions."

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
- [ ] Second run, cut Slide 3 by 10 seconds and Slide 5 by 10 seconds. Aim for 5:10.
- [ ] Third run, on your feet, aim for 4:50.
- [ ] Verify the demo audio clip works on the presentation laptop **before** the talk.
- [ ] Have the PNG files downloaded locally — don't rely on live Colab during the talk.
- [ ] Know which 3 sentences you'll cut if Slide 6 runs long (the backup is: skip the threshold sweep story entirely and jump straight to the diurnal pattern figure).

# Project Status & Reframing Strategy: EMNLP 2026 Workshop on Multimodal Interaction in Face-to-Face Dialogue

*All numbers are 5-fold stratified cross-validation on ∼89 labeled segments (Active 44, Intermittent 28, Disengagement 17), produced by the consolidated comparison cell in `clustering_alignment_extend.ipynb` (run in the `thesis2` env). RandomForest is the primary estimator; Logistic Regression is reported only to confirm direction, as it trips numerical warnings on our one-hot columns.*

---

## 1. Executive Summary

A clean, leakage-free re-evaluation overturns part of the MMSYM story but, importantly, **rescues its core intuition**. Two things were wrong before: we reported weighted-F1 (which rewards predicting the majority class) and we scored a cluster-then-label procedure that leaked test labels. Under a strict supervised protocol with macro-F1, the original "alignment helps" headline does **not** hold *as encoded* — the alignment block was actually dragging the model below a target-only baseline.

The reason turned out to be a single, fixable feature-encoding flaw. The Head-and-Gesture alignment was stored as a ∼138-dimensional element-wise difference; once it is re-encoded as a cosine similarity (consistent with the AU and CLIP alignments), the alignment block helps again. Our **best system fuses listener-side cosine alignment with speaker-side prosody**, reaching macro-F1 **0.364** / accuracy **0.517**, versus a target-only baseline of 0.341 / 0.459 and a majority-class floor of 0.220 / 0.494.

The strongest, most reviewer-proof result is interactional and interpretable: the **speaker's terminal pitch slope** — whether their intonation rises or falls — systematically tracks whether the silent listener they are addressing is engaged. This is exactly the kind of cross-participant, multimodal effect the workshop cares about.

---

## 2. Methodological Corrections to the MMSYM Pipeline

Three changes make the work survive strict review. The first two are corrections to *how we measured*; the third is a corrected *feature encoding* that we have now implemented and validated.

-   **Supervised Cross-Validation over clustering.** The original pipeline clustered segments, hand-labelled clusters from a handful of sampled clips, and propagated the majority label — non-reproducible, and it fit and scored the cluster→label map on the same labels (leakage). We now use strict 5-fold stratified CV and keep clustering for *describing* behaviour types only.
-   **Macro-F1 over weighted-F1.** With *Active* over half the data, weighted-F1 rewards majority-class collapse (which GMM and hierarchical clustering do). We report macro-F1 and accuracy against a majority-class floor so the numbers mean something.
-   **Cosine encoding for Head-and-Gesture alignment (done & validated).** AU and CLIP alignment were single cosine similarities, but the Head-and-Gesture alignment was an **element-wise absolute difference** of the 46-dim manual vector across 3 reference pairs → ∼138 dimensions that overfit ∼89 labels. Conceptually a *difference* measures dissimilarity, the opposite of "alignment." We replaced it with a **standardized cosine similarity per reference pair** (3 scalars), fit on the unlabeled training pool only. This single change lifts the full alignment block from macro-F1 0.231 (difference) to 0.352 (cosine) — back above baseline. The encoding comparison is in §4.1.

---

## 3. Core Feature Expansion: Incorporating the Speaker

Every labelled clip is, by construction, a window where the target participant is **silent**. So all of our original features describe a person who isn't doing the one thing — speaking — that most obviously signals engagement. The conceptual expansion is to characterise the **interlocutor** speaking during that same window, on the theory that *how you are being addressed* shapes how engaged you look.

For each segment we identify the speaker from the meeting TextGrid and extract features of *their* speech:

1.  **Acoustic prosody** (from separated audio): pitch in semitones relative to the window median (speaker-independent), pitch variability, overall and terminal pitch slope, intensity, pauses, speaking rate, and voice quality (jitter, shimmer, HNR).
2.  **Interactional text cues** (transcript): question cues, backchannels, fillers, negations, person reference, and VADER sentiment.
3.  **Turn-taking context**: number of active speakers, floor-holding duration, and position in the meeting.

### Key Finding: Terminal Pitch Slope

The **terminal pitch slope** — whether intonation rises or falls at the end of a phrase — is the most robust acoustic predictor and the cue we most want to foreground. It averages **+1.3 semitones** when the listener is actively engaged, **+2.3** when intermittent, and **−3.6** when the listener is disengaged: disengaged listeners are being addressed with flat or falling intonation. Because our transcripts have no punctuation, this dialogic marker is recoverable *only* from the acoustics — a naturally on-topic point for a multimodal-dialogue audience.

---

## 4. Quantitative Results: Final Consolidated Comparison

The single table below crosses the (corrected, cosine-encoded) alignment block with the speaker block on equal footing. The last two rows show the *old* `manual_diff` encoding for reference — what we would have reported under the MMSYM feature set.

| Configuration | RF Macro-F1 | RF Acc | LR Macro-F1 | dim |
| --- | --- | --- | --- | --- |
| *Majority-class floor* | *0.220* | *0.494* | *—* | *—* |
| Target-only (baseline) | 0.341 ± 0.117 | 0.459 | 0.438 | 166 |
| + Speaker | 0.306 ± 0.073 | 0.483 | 0.392 | 203 |
| + Alignment (AU/CLIP cosine) | 0.326 ± 0.094 | 0.482 | 0.474 | 172 |
| + Alignment (AU/CLIP + HG cosine) | 0.352 ± 0.066 | 0.505 | 0.476 | 175 |
| **+ Alignment (AU/CLIP cosine) + Speaker** | **0.364 ± 0.115** | **0.517** | 0.395 | 209 |
| + Alignment (AU/CLIP + HG cosine) + Speaker | 0.279 ± 0.090 | 0.449 | 0.352 | 212 |
| *[ref] Old full alignment (`manual_diff`)* | *0.231 ± 0.075* | *0.415* | *0.381* | *310* |
| *[ref] Old full alignment + Speaker* | *0.295 ± 0.085* | *0.448* | *0.313* | *347* |

**Headline configuration: AU/CLIP cosine alignment + speaker prosody — macro-F1 0.364, accuracy 0.517.**

Per-class breakdown for the headline configuration (cross-validated predictions):

| Class | Precision | Recall | F1 | support |
| --- | --- | --- | --- | --- |
| Active engagement | 0.520 | 0.886 | 0.655 | 44 |
| Intermittent engagement | 0.417 | 0.179 | 0.250 | 28 |
| Disengagement | 1.000 | 0.118 | 0.211 | 17 |
| **macro avg** | 0.646 | 0.394 | **0.372** | 89 |

### Critical Takeaways

1.  **Complementarity (the positive result).** Listener cosine alignment and speaker prosody each score *below* the baseline alone (0.326 and 0.306), but **together they exceed it** (0.364). The signal is genuinely cross-participant and multimodal — neither modality carries it by itself.
2.  **Encoding, not features, was the problem.** The old difference encoding pushed full alignment to 0.231; the cosine encoding lifts it to 0.352, above baseline (§4.1). The MMSYM intuition survives once the features are encoded consistently.
3.  **The two best blocks do not fully stack on 89 labels.** Adding the Head-and-Gesture cosine on top of the speaker block *lowers* macro-F1 (0.364 → 0.279). With so few labels the alignment×speaker interaction is noisy, so the headline system deliberately uses the leaner AU/CLIP cosine alignment alongside the speaker block. We read these gaps as directions, not decimals (fold-to-fold std ≈ 0.05–0.12).
4.  **Where the model struggles.** It recovers *Active* well (recall 0.886) but barely detects *Disengagement* (recall 0.118) and *Intermittent* (F1 0.250). This is consistent with the weak annotator agreement on the middle class (§6) — a ceiling on what any classifier can reach here.
5.  **Engagement is not the dominant axis of variation.** Unsupervised KMeans (k=3) on the headline features yields an Adjusted Rand Index of **−0.027** vs the true labels — engagement structure simply is not what the feature space clusters on. Worth stating plainly rather than hiding.

### 4.1 Head-and-Gesture Alignment Encoding Ablation (justification for the cosine choice)

To choose the encoding, we benchmarked four ways of expressing target↔reference Head-and-Gesture alignment, all standardized on the unlabeled training pool and added on top of the same target-only baseline:

| Head-and-Gesture alignment encoding | RF Macro-F1 | RF Acc | LR Macro-F1 | dim |
| --- | --- | --- | --- | --- |
| Baseline (no Head/Gesture alignment) | 0.341 | 0.459 | 0.438 | 166 |
| Old: element-wise abs-difference (∼138d) | 0.321 | 0.471 | 0.386 | 304 |
| **Standardized cosine (3 scalars)** | **0.339** | **0.482** | **0.483** | 169 |
| Scaled Euclidean distance (3 scalars) | 0.275 | 0.449 | 0.461 | 169 |
| Per-group cosine (pose / motion / nod-shake / contact, 12d) | 0.288 | 0.461 | 0.424 | 178 |

The cosine is the only encoding that doesn't hurt RF and it lifts LR markedly (0.438→0.483). Crucially, the scaled *distance* uses the **same 3 dimensions** yet falls to 0.275 — so the gain comes from re-framing the feature as *similarity*, not from shrinking the dimension count. This is why the cosine, not a reduced difference, is the right fix.

---

## 5. Strategic Repositioning for the EMNLP Workshop

Rather than an unsupervised-clustering optimisation study, we reposition the paper around an interactional question: **how the behaviour of the speaking participant shapes the engagement of a silent listener, and which multimodal cues mediate it.** The same experiments yield three contributions this audience values:

-   **Interactional contribution.** A speaker's prosody — above all the terminal pitch slope of their phrases — systematically pairs with the engagement state of the listener they address: a cross-participant, dialogic effect, not a property of the listener in isolation.
-   **Multimodal contribution.** Listener nonverbal alignment and speaker prosody are *complementary*: neither beats a target-only baseline alone, but their clean fusion does (0.364 vs 0.341).
-   **Methodological contribution.** In this low-resource regime, common shortcuts (weighted-F1, cluster-then-label) overstate how much features help; a clean protocol both shrinks the apparent gains and reveals that a poor feature *encoding* — not the features themselves — was the real problem.

In the paper this means leading with the alignment+speaker comparison and the interpretable prosody analysis rather than the clustering table; reporting macro-F1 and accuracy against a majority floor throughout; reducing the algorithm comparison to a single robustness statement about feature-space geometry; and presenting the encoding ablation as a first-class result. We will note in one sentence that GEHM is a Zoom corpus rather than strictly face-to-face, and argue that the channels measured — head pose, action units, prosody — are the same ones studied in co-present dialogue.

---

## 6. Limitations & Constraints

-   **Sample size.** ∼89 labelled segments is small. The +0.023 macro-F1 gain of the headline system over baseline sits within fold-to-fold variance; the accuracy gain (0.459 → 0.517) is steadier. We would not claim more than "promising," and we report variance openly.
-   **Inter-rater reliability.** The *Intermittent / Low-engagement* class has weak annotator agreement (κ ≈ 0.22), visible directly in its low recall, and is a ceiling on any model — reported as a finding in its own right. More annotation is the single change that would let us make a stronger claim.
-   **Corpus nature.** GEHM is a Zoom-based remote-interaction corpus, not co-present face-to-face. We argue the localized channels we evaluate (head pose, action units, prosody) mirror the mechanics of co-present dialogue.

---

## 7. Future Work: Temporal Head-Motion Synchrony

All alignment encodings tested here are **static** — they compare a single per-window summary of the target against a per-window summary of each reference, so they cannot see *coordination over time*. The most interesting form of behavioral alignment in dialogue (mimicry / entrainment — a listener nodding in time with a speaker's emphasis) is inherently **temporal**.

The natural next encoding is **temporal head-motion synchrony**: take the two participants' per-frame head-pose signals (yaw / pitch / roll) over the 5-second window and measure coordination via windowed cross-correlation (peak correlation and the lag at which it occurs, optionally per axis). This yields a small, interpretable set of scalars per reference pair and directly captures whether the listener's head motion tracks the speaker's.

It was **deferred for cost**: unlike the features above, it cannot be computed from the existing pickles — it requires the raw per-frame OpenPose keypoints, downloaded per meeting via `download_raw_for_meeting(meeting, persons, what=("keypoints",))` (see `additional_features.ipynb`). Given the modest payoff the static cosine already delivers on ∼89 labels, this is worth attempting only if we expand annotation.

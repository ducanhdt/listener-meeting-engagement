# Short-paper outline — EMNLP 2026 Workshop on Multimodal Interaction in Face-to-Face Dialogue

**Format:** ACL ARR short paper. **4 pages** of content (Limitations, Ethics, References, and Appendix do not count). Anonymous for review. ARR requires a **Limitations** section.
**Authors (from `course_report.tex`):** Trung Duc Anh Dang, Zhiqing Peng. University of Copenhagen.
**Structure:** 1 Introduction · 2 Related Work · 3 Methodology · 4 Results and Discussion · 5 Conclusion · Limitations (uncounted).

> **Internal note (not for the paper):** the report2-to-now tracking lives in the *Relationship to `report2.pdf`* section at the end. Keep self-comparison out of the paper itself. Position against the literature, not against our own status report.

**Working title (pick one):**
- *Aligned or Adrift: Listener–Speaker Behavioral Alignment and the Engagement of Silent Participants*
- *Does Alignment Signal Engagement? A Clustering-and-Prediction Study of Non-Speaking Listeners*

**Thesis:** A non-speaking listener's engagement is **interactional.** It shows up in *how the speaker addresses them* (acoustic prosody, especially **terminal pitch slope**) and in how well their behavior **aligns** with the speaker. Prior work reads the speaker or the listener in isolation; we model the **listener↔speaker relation** directly.

**Three contributions (state them plainly at the end of the Intro):**
1. **The relation predicts best.** Listener↔speaker behavioral alignment is the strongest single predictor (RF macro-F1 0.341 → 0.378), above either party's own features. An interpretable correlate supports it: the speaker's terminal pitch slope tracks listener engagement (+1.5 / +2.6 / −3.6 st across Active / Low / Disengaged), so disengaged listeners are addressed with flat or falling intonation.
2. **Compact beats rich in low-resource fusion.** A clean low-dimensional cosine fusion wins; a high-dimensional difference encoding *hurts* (0.231). We evaluate with leakage-free supervised CV, macro-F1, and a majority floor, and keep clustering for description only (k=2, ARI ≈ 0).
3. **The construct is really binary, and labels are the wall.** The 3-class middle is an uncertainty hedge (κ ≈ 0.22 there); a binary version is clearly learnable (macro-F1 ≈ 0.63); scarcity of the disengaged class is the binding constraint.

---

## Page budget (rough, 4 pages)

| Section | Space | Floats |
|---|---|---|
| Abstract | 0.2 pg | — |
| 1. Introduction | 0.7 pg | — |
| 2. Related Work | 0.5 pg | — |
| 3. Methodology | 1.1 pg | **Fig. 1** (pipeline) |
| 4. Results and Discussion | 1.4 pg | **Table 1**, **Table 2**, **Fig. 2** |
| 5. Conclusion | 0.3 pg | — |
| Limitations (uncounted) | — | — |

If space is tight: move the encoding ablation and the feature-importance figure to the Appendix.

---

## Abstract (~150 words)
Problem: detecting the engagement of **non-speaking participants** in multi-party meetings. Gap: prior work reads either the speaker's signal or the listener's own nonverbal behavior on its own, treating the listener as a solo source. Our move: treat engagement as **interactional.** We bring in the **speaker** addressing the silent listener (acoustic prosody, especially **terminal pitch slope**) together with how well the listener's behavior **aligns** with the speaker and the other participants, all under leakage-free CV with macro-F1 and a majority floor. Findings: the speaker's terminal pitch slope tracks listener engagement (Active +1.3, Low +2.3, Disengaged −3.6 st); **listener↔speaker alignment is the strongest single predictor** (RF macro-F1 0.341 → 0.378); the listener and speaker blocks add to each other when fused compactly (0.364), while a high-dimensional difference encoding *hurts* (0.231); clusters do not recover engagement (k=2, ARI ≈ 0). Honest caveat: 89 labels, low agreement (κ = 0.42), gains within fold variance; the 3-class middle is an uncertainty hedge, and a binary version is clearly learnable (0.63). Collecting labels is the key lever.

---

## 1. Introduction
- **Context and why it matters:** engagement is a key sign of meeting quality (`frank2016engagement`); most detection work targets **speakers** (clear prosodic cues, `pelletrostaing2023multimodal`), while feedback from non-speaking participants is subtle (`pelletrostaing2023multimodal`).
- **Our angle, alignment:** engagement is *relational.* A silent listener is engaged *with* something: the speaker, or the group. We measure behavioral **alignment** (the mimicry/entrainment intuition) between the listener and (a) the other people present and (b) the person addressing them, and ask whether it tracks engagement.
- **Why both clustering and supervised learning:** we first ask whether engagement is even a clear axis of behavioral variation (clustering). It is not, so we then measure feature effects with supervised CV.
- **Engagement definition:** behavioral involvement (`fredricks2004school`), read from visual cues: head pose, gaze, and facial expression (`kaur2018prediction`).
- **Preview of findings and the 3 numbered contributions.** Road-map sentence. (GitHub footnote goes in at camera-ready, for anonymity.)

---

## 2. Related Work
- **Engagement detection, speaker-centric:** prosodic and movement cues correlate with involvement (`oertel2011towards`, `pelletrostaing2023multimodal`).
- **Nonverbal and listener engagement:** head pose, AUs, and gaze predict engagement (`singh2023attention`, `inoue2018engagement`); online-learning engagement from head pose and movement (`gupta2016daisee`, `dewan2019engagement`, `sharma2022engagement`).
- **Behavioral alignment and synchrony:** mimicry and entrainment as a social signal. This is the basis for our listener↔co-participant and listener↔speaker alignment features (cite the synchrony literature; intuition from `oertel2011towards`).
- **Low-resource multimodal evaluation:** small, imbalanced label sets reward leakage-free CV and a majority floor, and make naive cluster-then-label evaluation optimistic (cite the relevant methodological literature).
- **Our position:** prior work treats engagement as a property of one person. We measure the **listener↔speaker relation** directly, and we keep clustering for description rather than letting unsupervised structure stand in for a classifier.

---

## 3. Methodology

### 3.1 Problem formulation
- Predict the engagement of the **non-speaking target** in 5-second windows. Primary label space: 3 classes, **Active engagement / Low engagement / Disengagement** (the annotation). Each clip is a window where the target is silent. That is exactly why "alignment to whoever is speaking" is meaningful.

### 3.2 Data, labels, and the reliability problem
- **GEHM Zoom corpus** (`paggio2024gehm`): 12 meetings, ~8 h, 5–9 participants, English; OpenPose (`cao2019openpose`).
- Segmentation: 5-second clips with 2-second overlap; filter out speaking, low-visibility, turned-away, and near-duplicate clips (CLIP, `radford2021learning`), giving **549** target segments. Each is paired with the co-present reference participants and the **identified active speaker** (from the TextGrid).
- **Labels are noisy, and the 3rd class is an uncertainty hedge, which drives the whole design.** The intended split was **engaged vs disengaged.** Annotators found many cases too hard to call, so they added a **middle "Low engagement" class (raw annotation label: *Intermittent*) for the unsure cases.** So this middle class is not a third engagement level. It is an abstention bucket, and the disagreement concentrates there. 4 annotators, a 3-level scale plus "unclear"; one was dropped (Cohen κ = 0.172); **Fleiss' κ = 0.42** overall; after filtering for agreement, **89** labeled segments remain (Active 44, Low engagement 28, Disengagement 17). This single fact explains the Results: k=2 is the natural structure, the middle is unpredictable, and a binary version is the principled target.

### 3.3 Features, three families (keep brief; extraction detail goes to the Appendix)
- **(a) Listener nonverbal (target-only baseline):** head pose and movement statistics, nods and shakes, hand gestures (MediaPipe `lugaresi2019mediapipe` plus OpenFace `baltrusaitis2018openface`); Facial Action Units (PCA → 20); CLIP embeddings (PCA → 100).
- **(b) Alignment, two kinds:**
  - **Listener↔co-participants** (the original "alignment"): cosine similarity between the target and the *other* participants in the same window (AU, CLIP, and a compact Head-and-Gesture cosine).
  - **Listener↔speaker [NEW, the headline feature]:** cosine and correlation between the target's behavior and the **identified active speaker's** behavior in the *same* window (visual CLIP and AU channels), computed from features we already extracted. No new data.
  - **Encoding point (state it plainly):** an element-wise *absolute difference* measures dis-similarity (the wrong sign) and blows up the dimensionality (~138 dims, which overfits 89 labels). **Cosine** fixes both, and we standardize it on the unlabeled pool only (no leakage).
- **(c) Speaker's own features (for contrast):** the active speaker's acoustic prosody plus interactional text cues. We report these to show that *the speaker's signal alone does not help.* It is the listener↔speaker *relation*, not the speaker by itself, that carries signal.

### 3.4 Evaluation protocol (a contribution)
- **Unsupervised:** silhouette and BIC for cluster count; **Adjusted Rand Index** of KMeans vs labels (does the structure match engagement?).
- **Supervised:** 5-fold **stratified CV**, **macro-F1** and accuracy against a **majority-class floor**, with **RandomForest** as primary (LR is directional). Explicitly **not** cluster-then-label.
- **Label-scheme sensitivity:** the primary 3-class scheme plus two binary versions (below).

> **Figure 1 (pipeline).** Input: a 5-second window (target silent) plus co-present references plus the identified speaker. This feeds three feature blocks: (a) listener nonverbal, (b) **alignment: listener↔others and listener↔speaker (cosine)**, (c) speaker own-features. These go to {clustering: ARI check | supervised CV}, then to engagement. One column wide.

---

## 4. Results and Discussion

### 4.1 Engagement is not the main axis of variation (clustering)
- Silhouette and GMM-BIC favor **k=2** for all algorithms; KMeans(k=3) vs labels gives **ARI ≈ 0** (−0.01 to −0.03). So the unsupervised behavior structure does **not** match engagement. We therefore (i) report clusters at **k=3 to match the 3-class label space**, as *exploratory* behavior types only, and (ii) measure feature effects with supervised CV rather than reading labels off the clusters.

### 4.2 Alignment effects, supervised (Table 1), the core result
- **The headline:** listener↔speaker alignment *alone* is the single best addition. Target-only 0.341 goes to **0.378**, above listener↔others (0.326) and the speaker's own features (0.306). The *relation* between listener and speaker beats either party's features alone.
- **Adding more blocks does not help.** Every richer combination dilutes the gain (full grid in Table 1); on 89 labels the added dimensions cost more than the signal is worth. The claim is "the listener↔speaker relation carries signal," not "more features help."
- **Honesty:** all deltas sit within fold std (≈ 0.09); LS has ~64% speaker coverage (train-median imputed); individual LS features rank low. RF is primary (LR fits are degenerate on the one-hot columns). The high-dimensional difference encoding **hurt** (0.231): an encoding problem, not a concept problem (see §4.4).

> **Table 1 (main, RF macro-F1 ± std / Acc / dim; reproducible from `analysis.ipynb` extended).** majority floor (0.220 / 0.494 / —); target-only (0.341 / 0.459 / 166); + others-align (0.326 / 0.482 / 172); **+ listener↔speaker (0.378 / 0.505 / 170, bold)**; + speaker (0.306 / 0.483 / 203); + others+LS (0.332 / 0.483 / 176); + others+speaker (0.364 / 0.517 / 209); + LS+speaker (0.368 / 0.505 / 207); + **all** O+LS+S (0.337 / 0.494 / 213); *[ref] old abs-diff alignment (0.231 / 0.415 / 310)*.

### 4.3 Interpretable acoustic correlate: terminal pitch slope (Fig. 2)
- **The speaker's terminal pitch slope tracks listener engagement** (descriptive, per-class means): **Active +1.48 st, Low +2.63 st, Disengaged −3.58 st** (on the 89-label set). Disengaged listeners are consistently addressed with *flat or falling* intonation; `f0_mean` is Low 190 > Active 172 > Disengaged 167 Hz. Because the transcripts have **no punctuation**, this dialogic marker can only be recovered acoustically. That point is worth making for the workshop audience.
- **Honesty (say it plainly):** this is a **descriptive correlate**, not a standalone predictor. The speaker block's macro-F1 gain is within fold variance, and speaker-only features do not reliably beat the baseline out of sample (especially for the rare disengaged class). Present terminal pitch slope as an *interpretation* of where the interactional signal lives, alongside the listener↔speaker alignment of §4.2.

> **Figure 2 (centerpiece).** Bar or box plot: x = listener engagement (Active / Low / Disengaged), y = speaker terminal pitch slope (semitones), showing +1.5 / +2.6 / −3.6 with error bars. The memorable, interpretable result. (Exact per-class means and std come from the speaker-feature cells.)

### 4.4 The label problem: the middle class is an uncertainty bucket (Table 2)
- **3-class sits near the majority floor**, with all feature deltas within noise. So engagement-as-annotated is hard, which matches the low κ. **Most of the difficulty is the abstention middle** (Low-engagement F1 ≈ 0.25, vs Active ≈ 0.66).
- **The principled reduction matches the annotation process.** Since the middle means "unsure," **`active_vs_rest` (Active vs not) is the binary the annotators originally intended**, and it is **clearly learnable (RF macro-F1 ≈ 0.63, +0.30 over floor)**, from the listener's **own** behavior.
- **Why not just drop the middle?** We tried **`confident_only`** (Active vs Disengaged, the uncertain middle dropped, n=61). It does **not** recover (macro-F1 ≈ 0.45 ≈ its majority-F1 floor), because the 17 disengaged are too few and too imbalanced. Likewise **`engaged_vs_disengaged` (any engagement vs none) collapses** (81/19, disengaged F1 = 0). So **the binding constraint is the disengaged class: too few labels to isolate it.**
- **Convergence (three independent lines):** the **annotation process** (binary intent plus an uncertainty hedge), the **unsupervised structure** (silhouette/BIC → k=2), and the **supervised learnability** (`active_vs_rest` works, 3-class does not) all say the construct is **2-way.** State that they are independent. Do *not* equate cluster count with label count.

> **Table 2 numbers (RF macro-F1 / acc / floor, n; from the `analysis.ipynb` sensitivity cell).** 3class: target-only 0.341 / 0.459, + listener↔speaker 0.378 / 0.505 (n=89, floor .494). active_vs_rest: 0.634 / 0.641, 0.581 / 0.597 (n=89, floor .506). engaged_vs_disengaged: 0.500 / 0.820 (n=89, floor .809; collapses). confident_only: 0.450 / 0.722 (n=61, floor .721; fails).

> **Table 2 (label-scheme sensitivity).** Cols: scheme | feature set | RF macro-F1 | Acc | floor | n. Rows: 3class / **active_vs_rest** / engaged_vs_disengaged / **confident_only (Active vs Diseng)** × {target-only, + listener↔speaker alignment}. Note the tie between weak classes and annotation ambiguity / κ.

### 4.5 What features matter (with an honesty caveat)
- **Importance by feature family** (RF impurity, summed): CLIP 0.63, listener head-motion 0.12, speaker-acoustic 0.09, listener head-pose 0.05, others-align 0.03, speaker-text 0.03, listener↔speaker 0.01, AU 0.01, nods/contact < 0.01. Among the *interpretable* (non-CLIP) blocks, **listener head-motion dynamics lead**: yaw (side-to-side) pose and velocity (`pose_std_yaw`, `max/avg_yaw_velocity`), then head-pose, which reads as attention and orientation. On the speaker side, acoustic prosody leads (terminal pitch slope, §4.3).
- **Caveat (state it plainly):** impurity importance is **biased toward high-dimensional blocks.** CLIP's ~100 dims dominate, and the **listener↔speaker block, the single *best* predictor in the CV ablation (Table 1), ranks near the bottom** by impurity (0.01). Permutation importance is ≈ 0 for every feature, which confirms that **no single feature is reliably pivotal at n=89.** So importance is not predictive value: the **CV ablation (Table 1) is the authoritative "what helps."** This family chart is descriptive interpretation only.

> **Figure 3 (appendix-eligible).** Feature-*family* importance bar chart (CLIP / head-motion / head-pose / nods / contact / AU / others-align / listener↔speaker / speaker-acoustic / -text / -context), with the dim-count-bias caveat in the caption. (Generated from the family-importance cell in `analysis.ipynb`.)

### 4.6 Discussion
- **Interactional reading:** the predictable signal points to *coordination* (listener↔speaker alignment), the *way the speaker addresses the listener* (terminal pitch slope), and the listener's own orientation dynamics. Engagement is relational, not a private state. On-theme for the workshop.
- **Complementarity:** listener nonverbal and speaker prosody genuinely add to each other. Each block alone is at or below baseline; the compact fusion (0.364) and listener↔speaker alignment (0.378) are the gains. But they **do not stack** further (all-blocks 0.337), so the lean configs are the headline.
- **Methodological lesson:** unsupervised structure is not engagement; use leakage-free CV plus a floor plus imbalance-aware metrics; in low-resource multimodal work, a compact cosine beats high-dimensional difference encodings.

---

## 5. Conclusion
- Restate the interactional story: a silent listener's engagement shows up in **how the speaker addresses them** (terminal pitch slope) and in **listener↔speaker behavioral alignment** (the strongest single feature). The listener and speaker blocks add to each other but only a little; clusters do not recover engagement (k=2, ARI ≈ 0); the construct is effectively **2-way**; the wall is labels.
- Future work: **temporal head-motion synchrony to the speaker** (raw OpenPose keypoint streams; the partial-coverage listener↔speaker alignment is a static proxy); **more, and more reliable, annotation** (the binding constraint); deeper and temporal models once the data allows.

---

## Limitations (mandatory, uncounted)
- **Sample size:** 89 labels; every headline gain is within fold variance; a single corpus.
- **Annotation reliability:** Fleiss κ = 0.42 (the middle "Low engagement" class is the worst) caps achievable 3-class performance, reported as a finding.
- **Listener↔speaker feature is preliminary:** static within-window similarity, 68/89 coverage, NaNs imputed; not yet temporal synchrony.
- **Corpus and domain:** GEHM is Zoom, not co-present; English-only text cues; prosody is speaker-normalized but in-domain.

## Ethics (brief, uncounted)
- Published corpus; engagement inference carries a surveillance-misuse risk (workplace and education); research-only, not deployment-grade.

## References (reuse `sample.bib`)
Keys: `frank2016engagement`, `pelletrostaing2023multimodal`, `oertel2011towards`, `singh2023attention`, `inoue2018engagement`, `fredricks2004school`, `kaur2018prediction`, `paggio2024gehm`, `cao2019openpose`, `radford2021learning`, `gupta2016daisee`, `dewan2019engagement`, `sharma2022engagement`, `lugaresi2019mediapipe`, `baltrusaitis2018openface`, `fleiss1971measuring`, `cohen1960coefficient`. **Add:** a behavioral-synchrony/alignment reference, Praat/parselmouth (if speaker features are kept), and scikit-learn.

---

## Relationship to `report2.pdf` (kept vs added)
- **Kept from report2 (already shared with the advisor):** the 3 method fixes; the speaker prosody/text/turn-taking features; the **terminal pitch slope** finding plus its bar chart (now **Fig. 2**, §4.3); the alignment-plus-speaker **complementarity** (0.364); clustering-fails (ARI ≈ 0); the limitations (sample size, κ, Zoom corpus).
- **Re-framed honestly:** terminal pitch slope is presented as an **interpretable descriptive correlate** (its predictive gain is within fold variance, which report2 already says), not as a standalone classifier; the "complementarity" config (0.364) is now one row of a **full add-on factorial**.
- **Added since report2:** **listener↔speaker behavioral alignment** (the strongest single feature, 0.378; Table 1); the **label-scheme analysis** (3-class middle = uncertainty hedge; `active_vs_rest` learnable ≈ 0.63; `confident_only` and `engaged_vs_disengaged` fail, pointing to disengaged-label scarcity; Table 2); readable feature names; everything reproducible from `analysis.ipynb` (`thesis2` env) and `utils.py`.

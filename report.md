# Where the project stands, and how I'd reframe it for the workshop

*For: supervisor review. Target venue: EMNLP 2026 Workshop on Multimodal Interaction in Face-to-Face Dialogue.*
*All numbers come from `clustering_alignment_extend.ipynb` (run in the `thesis2` env).*

## What I want to change about the MMSYM method

Looking back at the MMSYM pipeline, there are a few methodological choices I don't think I could defend in
front of workshop reviewers, and these are the things I want to change before we resubmit:

- **Clustering was doing the prediction.** We clustered the segments, hand-labelled each cluster by watching a few clips, and propagated the majority label. That isn't reproducible — it depends on which clips I happened to watch. I want to move to a clean supervised cross-validation and keep clustering only for *describing* behaviour types, not for scoring.
- **We reported weighted-F1.** With Active making up more than half the labels, weighted-F1 rewards a model that just leans on the majority class — exactly what GMM and hierarchical clustering do when they collapse — so it flattered the results. I want to switch to macro-F1 and accuracy reported against a majority-class floor.
- **The alignment features were encoded badly.** In the paper these come in two flavours: the AU and CLIP
  alignments are single cosine similarities, but the **Head-and-Gesture alignment** (head movement, nods, hand
  gestures) was computed as an **element-wise absolute difference**, which expands to ~138 dimensions and overfits
  on ~89 labels. And conceptually an absolute *difference* measures dissimilarity, which is backwards for
  something we call "alignment". I want to drop that encoding and **try several different ways of measuring
  Head-and-Gesture alignment** — a single cosine-similarity scalar to match the AU/CLIP features, a small set of
  interpretable measures (gesture co-occurrence, head-pose and movement-energy similarity), and, if the keypoints
  are worth downloading, temporal synchrony between the two participants' head-motion signals over the window.

## The new feature I want to introduce: the speaker

The bigger conceptual change is *what we add to the model*. Every labelled clip is, by construction, a window
where the target is **silent** — so all of our features so far describe a person who isn't doing the one thing
(speaking) that most obviously signals engagement. My idea is to characterise the person who is speaking during that same window, on the theory that how you are being addressed shapes how engaged you look.

For each segment I identify the speaker from the meeting TextGrid and extract features of *their* speech:

- **prosody** from their separated audio — pitch (in semitones relative to the window median, so it's
  speaker-independent), pitch variability, overall and terminal pitch slope, intensity, pauses, speaking rate,
  and voice quality (jitter / shimmer / HNR);
- **interactional text cues** from the transcript — question cues, backchannels, fillers, negations, person
  reference, and sentiment (VADER);
- a little **turn-taking context** — how many people are speaking, how long the speaker has held the floor, and
  position in the meeting.

The one that turned out to matter is the **terminal pitch slope** — whether the speaker's intonation rises or
falls at the end of a phrase. That's the cue I most want to foreground: it's the strongest predictor, it's
interpretable, and it's recoverable only from the acoustics, because our transcripts have no punctuation.

## The short version

I re-ran the evaluation more carefully, and I want to be honest about what came out of it, because it changes
what we can claim.

When I went back and built a *proper* baseline — target participant's own nonverbal features, with no alignment
and no leakage in the evaluation — the headline from our MMSYM abstract did not hold up. "Alignment features
improve engagement prediction" was, it turns out, partly an artifact of the metric we used (weighted-F1, which
rewards just predicting the majority class) and of the cluster-then-label procedure. Against a clean baseline,
adding the full alignment feature set actually *hurt* the classifier.

The good news is that the intuition behind the abstract survives once we fix the feature encoding. The reason
alignment looked harmful is that our "alignment" block secretly contained ~138 extra dimensions — the
element-wise absolute difference between the target's and the reference participants' Head-and-Gesture features —
and on only ~89 labeled segments that block just overfits. If we throw it away and keep the six interpretable
cosine-similarity features (plus the speaker-prosody features), alignment and speaker cues turn out to be
**complementary**: each is slightly
below baseline on its own, but together they beat it. So the story isn't "our features don't work" — it's "we had
the right idea but the wrong encoding, and we can only see that under a clean evaluation."

It's a more modest claim than the abstract made, but it's one I'm confident will survive a reviewer who runs their
own baseline — which the MMSYM version would not have.

## What the numbers actually say

Everything below is 5-fold stratified cross-validation on the ~89 labeled segments (Active 44, Intermittent 28,
Disengagement 17). I report macro-F1 as the primary metric because the classes are imbalanced, alongside accuracy
and a majority-class floor so the numbers mean something. RandomForest is the estimator I trust here; logistic
regression is in the notebook but it trips numerical warnings on our one-hot columns, so I wouldn't lean on it.

| configuration | RF macro-F1 | RF accuracy | dims |
|---|---|---|---|
| majority-class floor | 0.220 | 0.494 | — |
| baseline (target's own nonverbal features) | 0.341 ± 0.117 | 0.459 | 166 |
| + cosine-alignment (6 AU/CLIP similarities) | 0.326 ± 0.094 | 0.482 | 172 |
| + speaker prosody (no manual_diff) | 0.306 ± 0.073 | 0.483 | 203 |
| **+ cosine-alignment + speaker** | **0.364 ± 0.115** | **0.517** | 209 |
| + full alignment (Head-and-Gesture difference) | 0.231 ± 0.075 | 0.415 | 310 |
| + full alignment + speaker | 0.295 ± 0.085 | 0.448 | 347 |

Three things to take from this. First, the Head-and-Gesture difference block (`manual_diff` in the code) is what
was dragging everything down — every row that includes it falls apart. Second, alignment and speaker cues are genuinely complementary; the best
configuration is the compact one that combines them. Third, unsupervised clustering still never recovers the
engagement classes (KMeans k=3 gives an Adjusted Rand Index around −0.01 to −0.03), so engagement simply isn't a
dominant axis of variation in this feature space — that's worth saying plainly rather than hiding.

The one finding I find genuinely interesting, and that holds up regardless of the F1 numbers, is about the
*speaker*, not the listener. The speaker's terminal pitch slope tracks the listener's engagement: it averages
+1.3 semitones when the listener is actively engaged, +2.3 when intermittent, but **−3.6 when the listener is
disengaged**. Disengaged listeners are being spoken to with flat or falling intonation; engaged listeners get
rising, varied intonation. The speaker's mean pitch is the third most important feature out of 347. This matters
specifically because our transcripts have no punctuation, so "is this a question / is this engaging delivery" can
only come from the acoustics — which is a nicely on-topic point for a multimodal-dialogue audience.

## How the corrected results should change the paper

We should be upfront about the correction rather than quietly dropping the old claim — for a reviewed venue that
is both safer and more credible. The MMSYM weighted-F1 gain (0.31 → 0.40) does not reproduce under leakage-free
cross-validation with macro-F1; what we reported as a baseline was in fact the alignment configuration; and the
collapse of GMM and hierarchical clustering onto a single class is real. I'd present these not as failures but as
evidence: the collapse tells us something about how engagement does (and mostly does not) sit in the feature
space, and the metric correction is itself a useful warning for anyone doing low-resource multimodal
classification. The two fixes behind the change are simple — I removed a label leak in the cluster→label step
(the mapping was being fit and scored on the same labels), and I stopped reporting weighted-F1, which was masking
the single-class collapse.

## Repositioning the paper for the workshop

The MMSYM version was, at heart, a clustering study — it compared clustering algorithms and asked whether
alignment features improved a clustering-based classifier. For a workshop on multimodal interaction in dialogue,
I think the more compelling and better-fitting question is an interactional one: **how the behaviour of the person
speaking shapes the engagement of a silent listener, and which multimodal cues carry that signal.** The same
experiments then become three contributions this audience cares about:

- an **interaction finding** — a speaker's prosody, above all the terminal pitch slope of their phrases, relates
  systematically to whether the listener they are addressing is engaged; a genuinely cross-participant, dialogic
  effect rather than a property of the listener in isolation;
- a **multimodal finding** — listener-side nonverbal behaviour and speaker-side prosody are *complementary*:
  neither alone beats a target-only baseline, but together they do;
- a **methodological contribution** — in this low-resource regime, common evaluation shortcuts (weighted-F1,
  cluster-then-label) overstate how much the features help; a clean protocol both shrinks the apparent gains and
  reveals that a poor feature encoding, not the features themselves, was the real problem.

In the paper that means leading with the target → +alignment+speaker comparison and the interpretable prosody
analysis rather than the clustering table; reporting macro-F1 and accuracy against a majority-class floor
throughout; reducing the algorithm comparison to a single robustness table read as a statement about feature-space
geometry; using cluster inspection only to describe behaviour types, not to produce scores; and presenting the
encoding ablation as a first-class result. I'd also acknowledge in one sentence that GEHM is a Zoom corpus rather
than strictly face-to-face, and argue that the channels we measure — head pose, facial action units, prosody —
are the same ones studied in co-present dialogue.

## Limitations and what I'd like your steer on

The honest constraint is data. Eighty-nine labelled segments is small, the +0.023 macro-F1 improvement sits within
the fold-to-fold variance, and while the accuracy gain (0.46 → 0.52) is steadier I wouldn't claim more than
"promising." The intermittent/low-engagement label also has weak annotator agreement (κ ≈ 0.22), which puts a
ceiling on what any model can reach and is worth reporting as a finding in its own right. More annotation is the
single change that would let us make a stronger claim.

A few decisions I'd value your view on: whether to take this corrected version to the workshop (my recommendation)
or keep the MMSYM abstract as-is and treat this as a more rigorous follow-up; whether labelling more clips before
the deadline is realistic; and how much weight to put on the methodological angle versus the positive
complementarity result. Both are defensible — for this audience the interaction finding is the draw, and the
methodological point is probably our most novel contribution.

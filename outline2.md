# Short-Paper Outline: EMNLP 2026 Workshop on Multimodal Interaction

**Target Format:** ACL ARR Short Paper (4 pages max content, excluding Limitations/References)  
**Review Status:** Anonymous for Review  
**Working Title:** *Behavioral Alignment and Engagement Prediction for Non-Speaking Participants in Online Meetings*

---

## Abstract (~150 words)
* **Problem/Gap:** Detecting the engagement of non-speaking participants in multi-party meetings remains a challenge. Prior work predominantly treats the listener as an isolated signal source or relies heavily on speaker-centric cues. Furthermore, unsupervised approaches often conflate behavioral variance with true engagement states.
* **Methodology:** We frame listener engagement as inherently *interactional*. We evaluate the relationship between silent listeners and active speakers using two distinct lenses: (1) direct dyadic behavioral alignment (via low-dimensional cosine similarity), and (2) the prosodic context of the concurrently active speaker, treated as local interactional context that co-varies with listener engagement. Models are evaluated using a strict, leakage-free cross-validation protocol against a majority baseline floor.
* **Key Findings:** 1. Dyadic listener $\leftrightarrow$ speaker behavioral alignment is the single strongest predictor of engagement, improving Random Forest macro-$F_1$ from 0.341 to 0.378.
    2. The concurrent speaker's terminal pitch slope co-varies with listener engagement as an interpretable interactional correlate ($+1.5$ st for Active, $-3.6$ st for Disengaged).
    3. Multimodal fusion requires compact architectures; high-dimensional difference encodings collapse performance.
    4. Unsupervised clustering ($k=2$) fails to recover engagement labels ($\text{ARI} \approx 0$). The 3-class annotation middle operates as an uncertainty hedge; a binary active-vs-rest reduction yields a highly learnable signal ($\text{macro-}F_1 \approx 0.63$).

---

## 1. Introduction
* **Context:** Meeting quality heavily depends on participant engagement (frank2016engagement). While active speakers provide rich verbal and prosodic markers (pelletrostaing2023multimodal), feedback from non-speaking participants is exceptionally subtle.
* **The Paradigm Shift (Interactional Engagement):** **PP: I would not call this a paradigm shift. I'm not aware of it being a paradigm to start with!** We argue against treating the listener as a solo source. A silent listener's state is relational—manifested through behavioral alignment (mimicry/entrainment) with the group and through co-variation with the concurrently active speaker's nonverbal and prosodic behavior.
* **Methodological Tension:** We address whether engagement forms a natural axis of unsupervised behavioral variation. Finding it does not, we deploy supervised cross-validation to map specific interactive feature effects.
* **Core Contributions:**
    1. *Interaction Over Isolation:* Show that dyadic alignment outperforms individual-party features. Characterize the speaker's terminal pitch slope as a key communicative correlate.
    2. *Compact Multimodal Fusion:* Demonstrate that low-resource settings favor clean, low-dimensional cosine fusion over rich, high-dimensional encodings.
    3. *Construct Deconstruction:* Provide empirical evidence that the standard 3-class engagement construct is functionally binary due to annotation uncertainty hedges.

---

## 2. Related Work
* **Speaker-Centric Engagement:** Reviewing prosodic, acoustic, and positional markers of speaker involvement (oertel2011towards, pelletrostaing2023multimodal).
* **Non-Speaking Listener Behavior:** Prior computer vision approaches mapping head pose, gaze, and Facial Action Units (AUs) in isolated tracks (singh2023attention, inoue2018engagement, gupta2016daisee).
* **Behavioral Alignment & Synchrony:** Foundations of mimicry, entrainment, and social tracking as signals of group cohesion (oertel2011towards).
* **Positioning:** We explicitly close the gap between isolated listener tracking and social synchrony by evaluating the direct, windowed listener $\leftrightarrow$ speaker relation within a supervised framework, moving beyond purely descriptive or unvalidated clustering paradigms.

---

## 3. Methodology

[Window Input (5s clip)] ---> [Feature Blocks: Target Nonverbal | Alignment (Group & Dyadic) | Speaker Acoustic]
|---> Unsupervised Pipeline (Silhouette/BIC -> ARI Validation)
|---> Supervised Pipeline (5-Fold Leakage-Free CV -> Engagement Prediction)

### 3.1 Problem Formulation
* **Task:** Predict target listener engagement in 5-second non-overlapping windows where the target is entirely silent. 
* **Label Space:** Primary target is 3-class: *Active Engagement*, *Low Engagement* (annotated as Intermittent), and *Disengagement*.

### 3.2 Corpus & Annotation Reliability
* **Dataset:** GEHM Zoom corpus (paggio2024gehm); 12 remote multi-party meetings, ~8 hours, English. 
* **Data Preparation:** 549 valid silent segments mapped to co-present references and active speakers via TextGrid alignments. The active speaker for a window is operationalized as the co-participant with the largest speech overlap during that window; in multi-party meetings this speaker typically addresses the group rather than the specific silent listener, so we interpret speaker-side cues as concurrent interactional context.
* **The Reliability Constraint:** Label scarcity and noise shape our framework. Inter-rater agreement is modest (Fleiss' $\kappa = 0.42$). The middle "Low Engagement" class acts as an explicit coder hedge for ambiguous cases, concentrating the disagreement. Filtering for strict agreement yields 89 gold-labeled windows (44 Active, 28 Low, 17 Disengaged).

### 3.3 Feature Engineering
* **Family A (Listener Nonverbal Baseline):** Head pose statistics, velocities, micro-nods/shakes (baltrusaitis2018openface); PCA-reduced Action Units (20 dims) and CLIP visual embeddings (100 dims).
* **Family B (Alignment Features):**
    * *Listener $\leftrightarrow$ Co-participants:* Cosine similarity of target visual blocks against the mean of the rest of the group.
    * *Listener $\leftrightarrow$ Speaker:* Cosine similarity and cross-correlation directly between target features and the active speaker's visual streams within the same window.
* **Family C (Speaker Metrics):** Active speaker acoustic prosody (f0, pitch slopes) and structural conversational turn cues to isolate speaker-only effects.

### 3.4 Evaluation Architecture
* **Unsupervised:** Silhouette width and BIC to determine natural cluster numbers ($k$); Adjusted Rand Index (ARI) to check if natural clusters map onto true engagement.
* **Supervised:** Stratified 5-fold cross-validation. Models: Random Forest (primary, robust to high-dimensional blocks) and Logistic Regression. Metrics: Macro-$F_1$ and Accuracy compared against a strict majority floor. 

---

## 4. Results and Discussion

### 4.1 Unsupervised Structure Disconnect
* Silhouette and GMM-BIC metrics uniformly favor $k=2$ across feature spaces.
* Forcing $k=3$ to match annotations yields an $\text{ARI} \approx 0$ ($-0.01$ to $-0.03$). 
* *Takeaway:* Unsupervised behavioral clusters do not align with engagement axes. Clusters reflect stylistic behavioral types; supervised approaches are mandatory to isolate engagement.

### 4.2 Supervised Factorial Ablation (The Core Findings)

| Feature Configuration | Macro-$F_1$ $\pm$ std | Accuracy | Dimensions |
| :--- | :---: | :---: | :---: |
| *Baseline Floor (Majority)* | 0.220 | 0.494 | — |
| **Target Listener Only (Baseline)** | 0.341 $\pm$ 0.09 | 0.459 | 166 |
| + Group Alignment ($O$) | 0.326 | 0.482 | 172 |
| + **Speaker Alignment ($LS$) [Ours]** | **0.378** | **0.505** | **170** |
| + Speaker Metrics ($S$) | 0.306 | 0.483 | 203 |
| + Group ($O$) + Speaker ($S$) | 0.364 | 0.517 | 209 |
| + All Blocks Combined ($O + LS + S$) | 0.337 | 0.494 | 213 |
| *[Baseline Reference: High-Dim Diff Encoding]* | 0.231 | 0.415 | 310 | (consider to remove this row)

* **Analysis:** $LS$ alignment is the most effective single add-on. Richer, stacked feature sets expand dimensionality, diluting performance due to the sample size constraint ($n=89$). 
* *Methodological Lesson:* High-dimensional difference encodings collapse performance ($F_1 = 0.231$), proving that elegant, low-dimensional cosine metrics are essential for low-resource multimodal tasks.

### 4.3 Communicative Correlate: Speaker Terminal Pitch Slope
* Descriptive class means show a distinct vocal trend: windows with **Active** listeners co-occur with speaker rising intonation ($+1.5$ st), **Low-engagement** listeners with higher rises ($+2.6$ st), and **Disengaged** listeners with flat or falling tones ($-3.6$ st).
* *Interpretation:* This is a speaker-side prosodic correlate that co-varies with listener state, not a standalone predictor (its standalone supervised gain sits within fold variance). The speaker is identified by maximal speech overlap and generally addresses the group, we read this as local interactional context tracking the listener's state.

### 4.4 Resolving the Construct: Label-Scheme Sensitivity

| Scheme | Configuration | RF Macro-$F_1$ | Accuracy | Floor | $n$ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **3-Class** | Target + $LS$ | 0.378 | 0.505 | 0.494 | 89 |
| **Active vs. Rest** | Target + $LS$ | **0.634** | **0.641** | 0.506 | 89 |
| **Engaged vs. Disengaged** | Target + $LS$ | 0.500 | 0.820 | 0.809 | 89 |
| **Confident Only** (Drop Low) | Target + $LS$ | 0.450 | 0.722 | 0.721 | 61 |

* **Analysis:** Performance collapses when predicting the 3-class framework or isolating the sparse Disengaged class (due to severe class imbalance, e.g., 81 vs 19). 
* **The Convergence Argument:** Three lines of evidence prove the construct behaves binarily: (1) Annotator behavior (the middle class is an uncertainty hedge), (2) Unsupervised topography ($k=2$), and (3) Class learnability (Active vs Rest yields a robust $+0.30$ improvement over the baseline floor).

### 4.5 Feature Importance & Interpretation
* RF Impurity metrics are heavily skewed toward high-dimensional blocks (CLIP dominates at 0.63). 
* Among interpretable structural blocks, listener head-motion dynamics drive performance (yaw velocity and standard deviation), indicating orientation and attention patterns. 

---

## 5. Conclusion & Future Work
* **Summary:** Listener engagement is fundamentally relational. It is highly learnable when framed as a binary task, powered by dyadic behavioral alignment and specific speaker prosodic adaptations (terminal pitch slope).
* **Future Directions:** Transition from static window-based similarities to dense temporal tracking (e.g., synchrony over raw OpenPose coordinate streams) and expanding annotation pools to mitigate class scarcity.

---

## Limitations & Ethics (Uncounted Sections)
* **Limitations:** Small gold-label footprint ($n=89$); gains sit within cross-validation fold variance; high-dimensional impurity biases; interaction features currently act as window-level proxies rather than continuous temporal architectures.
* **Ethics:** Sourced from an open, peer-reviewed dataset. We explicitly highlight the potential dual-use risks regarding automated surveillance in workplace or educational software.

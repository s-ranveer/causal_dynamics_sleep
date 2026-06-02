## Dynamic Structural Causal Modeling for Sleep
This is our repository containing the supplementary for the work submitted to DDDAS 2026.

### Windowed Variables (process_data.py)
The windowed features computed for dynamic causal modelling are described below
  1. ApneaHypopnea (AHE/A). The burden of complete or near-complete upper airway obstruction events within the analysis window. Detected from the flow signal as drops of ≥90% (apnea) or ≥30% (hypopnea) from a 120s rolling median baseline, confirmed by accompanying oxygen desaturation of ≥3%. 
Ref: AASM Scoring Manual v2.6. We underestimate the AHI prevalence in a home sleep study, as EEG arousal information is not available in a Type IV Sleep Study.
  2. FlowLimitation (IFL/I). The burden of partial upper airway collapse within the analysis window is below the threshold for hypopnea scoring. Detected from the nasal pressure signal via the flattening index of individual inspiratory flow waveforms, requiring ≥4 consecutive flow-limited breaths not meeting hypopnea criteria. Ref:  Guevarra et al., Ann Am Thorac Soc (2022). We compute an approximation here as well.
  3. Desaturation (DeSat/D). The cumulative nocturnal hypoxic burden within the analysis window is independent of whether the underlying respiratory event meets apnea or hypopnea criteria. Measured as the fraction of the window occupied by SpO₂ drops of ≥3% from a 120s rolling mean baseline, sustained for ≥10 seconds. Ref: Temirbekov et al., Turk Arch Otorhinolaryngol (2018)
  4. FlowLimitedRespiratoryEffort (FLRE). We compute a 1 Hz effort signal and compare it to a rolling 120 s baseline. Flag sustained periods (≥10 s) with inspiratory flow limitation, elevated and rising effort, and no concurrent apnea/hypopnea or desaturation event. Retain events followed by either flow recovery or a pulse increase within 15 s. The final feature is the fraction of window time occupied by these flow-limited effort events. Ideally, we would have wanted to compute RERA; however, we can’t compute it only using the HSAT data
  5. Snoring (S). The burden of active snoring within the analysis window. Detected as bouts of sustained above-threshold energy on the snoring channel, with an adaptive threshold derived from the signal distribution, a minimum bout duration of ≥1s, and gap-merging for inter-snore silences of <2s. Ref: Maimon et al., J Clin Sleep Med. (2010).
  6. PulseActivation (Pulse/P). The burden of short-lived autonomic activations within the analysis window reflects sympathetic surges following respiratory events. Serves as a cardiovascular surrogate for arousal in the absence of EEG data. Detected as transient elevations of pulse rate of ≥5 bpm above a 30s rolling baseline, with a non-negative pulse trend, lasting ≥3s, with gap-merging for brief intervening drops.

**Note**

The variables are not referred to as such in the code and are referred to as
  
  |Feature|Name in Code|Node on Graph|
  |-------|------------|-------------|
  |AHE|ahe_proxy_fraction|A|
  |IFL|ifl_proxy_breath_fraction|I|
  |FLRE|flow_limited_effort_fraction|F|
  |Pulse|pulse_activation_fraction|P|
  |DeSat|odi3_desaturation_fraction|D|
  |Snoring|snoring_bout_fraction|S|


## Learning DSCMS (learn_dscms.py)
We are using the tigramite package to learn the PCMCI+ model from the data. We would be learning the time-series causal graph across different bootstrap samples and merging them using model averaging to get the final time-series causal graph. Please look into the file on how to set the hyperparameters such as the number of bootstraps, mci, and posterior threshold.

## Results 
The results are presented in the results directory

## Remaining Code and Data
The data is not available at the moment, and its availability can't be guaranteed in the future. The full code will be made available once the 

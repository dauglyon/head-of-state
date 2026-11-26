# Prior work

Compiled 2026-11-25 from a literature survey (2019 to November 2026). Each entry was read at the linked
primary source unless marked "abstract only"; figures are as reported by the authors. The survey
found no published method that combines zero-shot per-token regime scores from a language model with
an HMM-like, change-point or semi-Markov decoder.

## Reading a label from a frozen language model

- **Option-letter readout with calibration.** Contextual calibration (Zhao et al., 2021) divides out the
  label probabilities the model gives a content-free input: GPT-3 175B on AGNews 0-shot rose from 43.9
  to 73.9, but it sometimes hurts (GPT-2 on CB: 44.6 to 17.9). https://ar5iv.labs.arxiv.org/html/2102.09690
  Domain-conditional PMI (Holtzman et al., 2021) scores options against a short task string.
  https://ar5iv.labs.arxiv.org/html/2104.08315 Batch calibration (Zhou et al., 2023) subtracts the mean
  prediction over unlabelled inputs. https://arxiv.org/html/2309.17249
- **Option-order and format sensitivity.** Reordering options moves accuracy substantially
  (Pezeshkpour & Hruschka, 2023, abstract only, https://arxiv.org/abs/2308.11483); for instruction-tuned
  models the option-letter probability disagrees with the generated answer in up to over 60% of cases
  (Wang et al., 2024, https://arxiv.org/abs/2402.14499).
- **Scoring the text given each label ("channel" or generative classification).** Noisy-channel
  prompting (Min et al., 2022) scores P(text | label) and was more stable than direct scoring on GPT-2
  (https://ar5iv.labs.arxiv.org/html/2108.04106); Gen-Z (Kumar et al., 2024) scores the text under
  paraphrased label descriptions (https://arxiv.org/html/2311.07115). GeDi (Krause et al., 2020) updates
  a per-token class posterior from class-conditioned next-token probabilities, but with a fine-tuned
  conditional model (https://ar5iv.labs.arxiv.org/html/2009.06367).
- **Representation readouts.** Similarity between text and label-description embeddings
  (Schopf et al., 2022, abstract only, https://arxiv.org/abs/2211.16285); Patchscopes decodes a hidden
  state through a second prompt (https://arxiv.org/html/2401.06102v4).

## Segmentation and boundary signals

- **Surprisal boundaries.** EM-LLM marks a boundary where a token's surprisal exceeds the recent mean by
  γ standard deviations, at no extra pass (https://arxiv.org/html/2407.09450v3). Perplexity-based
  chunking (Meta-Chunking, 2024) cuts at perplexity extrema (https://arxiv.org/html/2410.12788v3).
- **Whole-document LLM segmentation with a run-time label list.** POSR (Wang et al., 2024) prompts once
  over a transcript and returns labelled segments (P_k 0.13 with a frontier model vs about 0.58 for
  TextTiling); not online (https://arxiv.org/html/2411.07598).
- **Zero-shot sequential sentence classification.** Gemma-2B 0-shot on PubMed 20k RCT: micro-F1 0.171
  (vs 0.712 at 5-shot) (https://arxiv.org/html/2411.15623v2).

## Online decoders

- **CUSUM** is optimal under Lorden's criterion; worst-case detection delay ≈ |log α| / KL, where α is the
  false-alarm rate and KL the divergence between the evidence before and after the change
  (Veeravalli & Banerjee, 2012, https://ar5iv.labs.arxiv.org/html/1210.5552).
- **Bayesian online changepoint detection** (Adams & MacKay, 2007) tracks a posterior over the current
  run length (https://ar5iv.arxiv.org/html/0710.3742).
- **On-line Viterbi** (Šrámek et al., 2007) emits the decoded path as the stream arrives (search
  summary of the abstract, https://arxiv.org/pdf/0704.0062). Explicit-duration semi-Markov models
  (Johnson & Willsky, 2012, abstract only, https://arxiv.org/abs/1203.1365).

## Lag after a switch

- Earlier context shifts a model's output for a median of about 87–168 tokens after an unrelated
  earlier exchange (Wang et al., IJCAI 2026, https://ijcai.org/proceedings/2026/88); measured on
  generation, not on classifying a context.
- Zero-shot detection of topic pivots in spliced dialogues: smaller open models carry stale context at
  over 0.75 of true pivots (ICLR 2026 workshop, https://arxiv.org/html/2605.09268).

## Efficiency

Shared-prefix attention and batched branches (Hydragen, https://arxiv.org/abs/2402.05099; SGLang,
https://arxiv.org/abs/2312.07104; Parallel Prompting, https://arxiv.org/abs/2611.15130) make many
questions over one cached context cheaper; none was used here.

## Not verified from a primary source

EM-LLM's trailing-window size; the Bayesian-surprise formula of Kumar et al. (2023, page blocked); PriDe
figures; Mind Your Format channel-versus-direct figures; HSMM inference cost; On-line Viterbi details
beyond the abstract.

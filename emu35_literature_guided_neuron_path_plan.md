# Emu3.5 Literature-Guided Neuron / Path Plan

## Purpose

This note maps the additional papers to concrete Emu3.5 experiments.
The goal is not to copy diffusion cross-attention analyses directly, because Emu3.5 is a token-based decoder-only UMM.
The goal is to use these papers to sharpen the current question:

```text
During generation or understanding, do internal neurons / heads / feature directions form a causal activation path?
Can we find, confirm, and block that path?
```

## 1. Architectural Translation

Diffusion T2I papers often study:

```text
text encoder -> cross-attention -> UNet / DiT -> image
```

For Emu3.5, the corresponding path is:

```text
text/image tokens
-> self-attention / MLP / residual stream
-> text-token logits or image-token logits
```

Therefore, for this project:

| external concept | Emu3.5 analogue |
|---|---|
| diffusion cross-attention layer | decoder self-attention edge / head |
| UNet/DiT generation block | decoder layer at image-token generation positions |
| concept neuron pruning | MLP intermediate neuron / feature direction masking |
| prompt word to image region attention | prompt token to generated image-token self-attention / residual effect |
| image-to-text gate | image / EOI-like token to answer-position residual/head route |

## 2. Literature Mapping

| paper / method | what to borrow | Emu3.5 experiment |
|---|---|---|
| Narrow Gate | image-to-text gate, EOI bottleneck, attention knockout | understanding image/EOI -> answer-position knockout; compare with L62 heads 36/38/46 |
| BLIP Causal Tracing | VLM clean/corrupt/patch template | understanding clean/corrupt image-color patch recovery |
| FCCT object tracing | visual/text token + MHSA/FFN/hidden-state causal map | fine-grained object/color/relation path tracing across decoder layers |
| V-SEAM | semantic perturbations: object / attribute / relation | matched perturbation sets: red->blue, object swap, left->right |
| AIA | U/G task-specific cross-modal interaction motivation | use differential score `S_gen - lambda S_under` |
| LocoGen | mechanistic localization of generation attributes | generation direct-effect intervention on heads/MLP/residual states |
| Prompt-to-Prompt | prompt token controls image spatial/semantic content | prompt-token to image-token self-attention/path patch |
| Attend-and-Excite | prompt-following / binding metrics | evaluate object neglect, color binding, relation errors after path masking |
| Textual generation localization | subskill-specific generation circuit | search paths for text rendering / color binding / count separately |
| ConceptPrune | skilled neuron pruning / zero-out | MLP neuron/feature masking for generation-specific color signal |
| Dictionary learning / SAE for T2I | features > raw neurons under superposition | train/probe SAE or PCA features on Emu3.5 MLP activations |
| ROME causal tracing | clean/corrupt/restore and MLP causal centers | layer x component causal traces for generation and understanding |
| Best Practices of Activation Patching | metrics, baselines, corruption design | add random/wrong/object controls and report decoded + likelihood metrics |
| ACDC / path patching | graph-level circuit discovery | move from component ranking to edge/path graph pruning |

## 3. Current Evidence Interpreted Through These Papers

### Generation

Current evidence:

```text
L61 visual-score MLP neurons are rankable by clean/corrupt activation delta.
top10/top50 are weak.
top1000 gives +0.127074 recovery.
PCA200 gives +0.169549 recovery.
full MLP module gives +0.246532.
full residual gives about +0.999.
```

Interpretation:

```text
This matches the superposition / dictionary-learning expectation:
the generation signal is not sparse in raw neurons,
but appears as broad MLP feature mass and low-rank feature directions.
```

Stage 45-46 add prompt-side decoded evidence:

```text
color-token input embedding patch: 2/4 exact, 4/4 improved.
object-token input embedding control: 0/4 exact, 0/4 improved.
```

Interpretation:

```text
The prompt color token is a compact input boundary.
The internal neuron path downstream is still unresolved.
```

### Understanding

Current evidence:

```text
L61-L63 answer-score residual patch is near full recovery.
post-image text/question residual states mediate image information.
L62 heads 36/38/46 recover understanding and are near-zero on generation.
```

Interpretation:

```text
This is closest to Narrow Gate / VLM causal tracing:
understanding has a late residual/head-level route.
It is not yet sparse-neuron evidence.
```

## 4. Revised Experimental Roadmap

### Stage 47: Wrong / Random Color-Token Decoded Controls

Purpose:

```text
Strengthen Stage 45 by testing whether the decoded color-token effect is genuinely color-specific.
```

Design:

```text
clean source: correct color-token embedding
wrong source: corrupt/wrong color-token embedding
random source: color token from another pair
object control: already Stage 46
```

Metric:

```text
exact HSV restoration
hue-distance improvement
object preservation
```

Allowed claim if positive:

```text
Prompt color-token input boundary has controlled decoded influence on generation hue.
```

### Stage 48: Neuron Activation Law For Generation

Purpose:

```text
Return from token boundary to neuron/feature activation behavior.
```

Design:

```text
collect L60-L63 MLP intermediate activations
group by generation visual-score positions
rank neurons by clean/corrupt activation delta
plot cumulative top-k recovery: 10, 50, 100, 200, 500, 1000, 2000, 5000
compare PCA / random / bottom-k
compare transfer to understanding score positions
```

Visualization:

```text
layer x top-k recovery heatmap
neuron-rank cumulative curve
activation delta distribution
PCA rank curve
```

Allowed claim:

```text
generation signal follows a broad feature-mass / low-rank-subspace activation law.
```

Not allowed:

```text
sparse generation neuron unless small k approaches module-level recovery and beats controls.
```

### Stage 49: Feature Masking / Projection-Out

Purpose:

```text
Move from confirm-by-patching to block-by-masking.
```

Design:

```text
identify generation PCA directions or top-k neuron mass at L61 visual-score MLP
run clean generation / teacher-forced scoring
mask selected neurons or project out selected PCA direction
measure loss of clean visual likelihood and decoded hue quality
```

Metric:

```text
clean target NLL increase
decoded hue failure
object preservation
cross-task understanding effect
```

This follows the ConceptPrune / SAE idea more than raw activation patching.

### Stage 50: Understanding Head Knockout

Purpose:

```text
Block the known understanding path.
```

Design:

```text
clean understanding examples
knockout L62 heads 36/38/46
compare answer-token NLL / answer accuracy
random-head and generation-head controls
```

Allowed claim:

```text
L62 heads 36/38/46 are not just recoverable under patching,
but necessary for part of understanding behavior.
```

### Stage 51: Edge / Path Patch Graph

Purpose:

```text
Move from component localization to path-level circuit discovery.
```

Design:

```text
nodes: prompt color token, image/EOI tokens, heads, MLP features, residual positions
edges: attention head routes and residual/MLP transitions
method: ACDC-style edge pruning / path patching
```

Generation candidate graph:

```text
prompt color-token input
-> selected early/mid self-attn heads
-> L61 MLP feature mass / PCA subspace
-> visual-token logits
```

Understanding candidate graph:

```text
image/EOI-like token
-> post-image text/question residual states
-> L62 heads 36/38/46
-> L61-L63 answer-score residual
-> answer logits
```

## 5. Near-Term Priority

The next two experiments should be:

```text
Stage 47: decoded wrong/random color-token controls
Stage 48: generation MLP neuron activation-law visualization
```

Reason:

```text
Stage 47 protects the compact color-token path claim.
Stage 48 returns the project to the user's core question: neuron activation behavior.
```

## 6. References

- The Narrow Gate: Localized Image-Text Communication in Vision-Language Models.
- Towards Vision-Language Mechanistic Interpretability: A Causal Tracing Tool for BLIP.
- Causal Tracing of Object Representations in Large Vision Language Models.
- V-SEAM: Visual Semantic Editing and Attention Modulating for Causal Interpretability of Vision-Language Models.
- On Mechanistic Knowledge Localization in Text-to-Image Generative Models / LocoGen.
- Prompt-to-Prompt Image Editing with Cross-Attention Control.
- Attend-and-Excite: Attention-Based Semantic Guidance for Text-to-Image Diffusion Models.
- ConceptPrune: Concept Editing in Diffusion Models via Skilled Neuron Pruning.
- Interpreting Large Text-to-Image Diffusion Models with Dictionary Learning.
- Locating and Editing Factual Associations in GPT / ROME.
- Towards Best Practices of Activation Patching in Language Models.
- Towards Automated Circuit Discovery for Mechanistic Interpretability.

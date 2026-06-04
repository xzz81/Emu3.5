# Emu3.5 Neuron / Feature Activation Path Visualization

## 1. What Counts As A Neuron Path Here

In this investigation, an MLP "neuron" means one intermediate channel in the decoder MLP:

```text
residual h_l
-> RMSNorm
-> gate_proj / up_proj
-> activation neuron z_i = SiLU(gate_i) * up_i
-> down_proj contribution
-> residual h_{l+1}
-> later layers
-> logits / decoded image
```

The token position is only the observation site, for example "visual-score positions".
The path itself is inside the model.

## 2. Generation Neuron / Feature Path

```mermaid
flowchart LR
    H60["Residual stream\nL60 visual-score state"]
    A61["L61 self-attn heads\nTop10 heads\n+0.101426"]
    M61["L61 MLP module\n+0.246532"]
    Z61["L61 MLP intermediate\n25,600 neurons"]
    K10["top10 neurons\n+0.036203"]
    K50["top50 neurons\n+0.045597"]
    K1000["top1000 neurons\n+0.127074"]
    PCA200["PCA200 feature directions\n+0.169549"]
    DP["MLP down_proj\nfeature contribution"]
    H62["Residual stream\nL61/L62 output"]
    LOGITS["visual-token logits"]
    IMG["decoded hue"]

    H60 --> A61 --> H62
    H60 --> M61 --> Z61
    Z61 --> K10 --> DP
    Z61 --> K50 --> DP
    Z61 --> K1000 --> DP
    Z61 --> PCA200 --> DP
    DP --> H62 --> LOGITS --> IMG

    H60 -. "full residual\n~+0.999" .-> H62
    M61 -. "module upper bound\n+0.246532" .-> H62

    K10 -. "too weak for sparse neuron claim" .-> IMG
    K1000 -. "broad feature mass" .-> IMG
    PCA200 -. "structured feature subspace" .-> IMG
```

## 3. Generation Evidence As A Ranked Activation Structure

```mermaid
flowchart TD
    OBS["Generation observation site:\nL61 visual-score positions"]
    SORT["Rank MLP neurons by\nclean/corrupt activation delta"]
    SMALL["Small sparse set\nTop10 / Top50"]
    MASS["Broad neuron mass\nTop1000"]
    SUBSPACE["Low-rank feature subspace\nPCA100 / PCA200"]
    MODULE["Full MLP module"]
    RESID["Full residual trajectory"]

    OBS --> SORT
    SORT --> SMALL
    SORT --> MASS
    SORT --> SUBSPACE
    SORT --> MODULE
    MODULE --> RESID

    SMALL --> S1["+0.036 / +0.046\nweak"]
    MASS --> S2["+0.127\nreal but broad"]
    SUBSPACE --> S3["PCA100 +0.129\nPCA200 +0.170"]
    MODULE --> S4["+0.247"]
    RESID --> S5["~+0.999\nnot localized"]
```

## 4. Why This Is Not A Single-Neuron Path

```mermaid
flowchart TD
    CLAIM["Can we say one generation neuron?"]
    T10["top10 neurons\n+0.036203"]
    T50["top50 neurons\n+0.045597"]
    T1000["top1000 neurons\n+0.127074"]
    PCA["PCA200 directions\n+0.169549"]
    MLP["full MLP module\n+0.246532"]
    ANSWER["No sparse-neuron claim yet"]

    CLAIM --> T10 --> ANSWER
    CLAIM --> T50 --> ANSWER
    CLAIM --> T1000
    CLAIM --> PCA
    T1000 --> DIST["distributed feature mass"]
    PCA --> FEAT["structured feature subspace"]
    DIST --> MLP
    FEAT --> MLP
```

## 5. Understanding Activation Path

```mermaid
flowchart LR
    IMG["Image-token residual states"]
    TXT["Post-image text/question\nresidual states"]
    H62["L62 attention heads\n36 / 38 / 46\nU-specific"]
    R61["Late residual stream\nL61-L63 answer-score positions"]
    LOGIT["answer-token logits"]

    IMG --> TXT --> R61 --> LOGIT
    IMG --> H62 --> R61

    R61 -. "residual patch:\nnear full recovery" .-> LOGIT
    H62 -. "head patch:\n+0.152832 understanding\n+0.001438 generation" .-> LOGIT
```

## 6. Current Path-Level Claims

```mermaid
flowchart TD
    G["Generation"]
    U["Understanding"]

    G --> GPATH["L61 visual-score MLP feature mass / PCA subspace"]
    G --> GHEAD["L61 target-side attention heads\npartial"]
    G --> GINPUT["Prompt color-token input embedding\ncompact decoded partial control"]
    G --> GNO["No sparse generation neuron yet"]

    U --> UPATH["Late residual answer-score route"]
    U --> UHEAD["L62 heads 36/38/46"]
    U --> UNO["No sparse understanding neuron yet"]
```

## 7. What To Visualize Next

The next neuron-level visualization should not be another token diagram.
It should plot:

```text
Layer x neuron rank x activation delta
Layer x cumulative top-k recovery
Layer x PCA feature recovery
Clean vs corrupt activation distribution for top generation neurons
Cross-task transfer: generation-selected neurons on understanding examples
```

These would turn the current pathway evidence into an explicit neuron activation map.

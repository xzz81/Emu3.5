# Emu3.5 Generation / Understanding Path Visualization

## 1. Current Evidence Map

```mermaid
flowchart TD
    Q["Original question:\nAre there generation / understanding neurons?"]
    Q --> A["Activation screening\nTask-differential candidates"]
    A --> B["Causal patching\nclean/corrupt recovery"]
    B --> C["Pathway evidence"]
    B --> N["Sparse neuron evidence?"]

    N --> N1["Generation sparse neurons:\nnot supported"]
    N --> N2["Understanding sparse neurons:\nnot supported"]

    C --> G["Generation pathway"]
    C --> U["Understanding pathway"]

    G --> G1["Target-side full-state trajectory overwrite\nStrong decoded restoration"]
    G --> G2["Prompt color-token input embedding\nCompact decoded partial control"]
    G --> G3["L61 MLP feature mass / PCA subspace\nTeacher-forced likelihood signal"]

    U --> U1["Late answer-score residual stream\nStrong recovery"]
    U --> U2["Post-image text/question residual mediation"]
    U --> U3["L62 heads 36/38/46\nHead-level route"]
```

## 2. Generation: What We Have Actually Found

```mermaid
flowchart LR
    P["Prompt color token\ncyan / magenta / red / purple"]
    E["Input embedding\n1-2 prompt positions"]
    D["Decoder computation"]
    V["Visual token trajectory"]
    I["Decoded image hue"]

    P --> E --> D --> V --> I

    E -. "Stage 44\nteacher-forced recovery +1.012833" .-> V
    E -. "Stage 45\n2/4 exact, 4/4 improved" .-> I

    O["Object token embedding\n1 prompt position"]
    O -. "Stage 46 control\n0/4 exact, 0/4 improved" .-> I

    T["Target-side input/residual states\n1024 visual-score positions"]
    T -. "Stage 24-43\n3/4 exact, 4/4 improved\nbut trajectory overwrite" .-> I
```

## 3. Generation: Neuron / Feature Interpretation

```mermaid
flowchart TD
    L61["L61 visual-score MLP intermediate\n25,600 neurons"]

    L61 --> S10["top10 neurons\n+0.036203\nweak"]
    L61 --> S50["top50 neurons\n+0.045597\nweak"]
    L61 --> S1000["top1000 neurons\n+0.127074\nreal signal"]
    L61 --> PCA["PCA200 directions\n+0.169549\nmore compressed"]
    L61 --> MLP["full MLP module\n+0.246532\nmodule upper bound"]
    L61 --> RES["full residual\n~+0.999\ntrajectory-level upper bound"]

    S10 --> C1["Not sparse-neuron evidence"]
    S50 --> C1
    S1000 --> C2["Supports broad feature mass"]
    PCA --> C3["Supports structured feature subspace"]
    MLP --> C4["MLP contributes, but not enough for decoded full restoration"]
    RES --> C5["Strong but too broad:\nnot localized neuron/circuit"]
```

## 4. Understanding Pathway

```mermaid
flowchart LR
    IMG["Image tokens"]
    TXT["Post-image text/question residual states"]
    H["L62 heads 36/38/46"]
    R["Late residual stream\nL61-L63 answer-score positions"]
    ANS["Answer token likelihood"]

    IMG --> TXT --> R --> ANS
    IMG --> H --> R

    R -. "Residual patch:\nnear full recovery" .-> ANS
    H -. "Head patch:\nU-specific recovery" .-> ANS

    GCTRL["Generation control"]
    H -. "near zero on generation" .-> GCTRL
```

## 5. Claim Boundary

```mermaid
flowchart TD
    YES["Can claim"]
    NO["Cannot claim yet"]

    YES --> Y1["Task-differentiated causal pathways"]
    YES --> Y2["Understanding: late residual + L62 head route"]
    YES --> Y3["Generation: prompt color-token input has decoded causal influence"]
    YES --> Y4["Generation: L61 MLP has broad feature mass / PCA subspace"]

    NO --> N1["Sparse generation neurons"]
    NO --> N2["Sparse understanding neurons"]
    NO --> N3["Complete compact generation circuit"]
    NO --> N4["Full residual patch equals localized mechanism"]
```

## 6. Next Experiments

```mermaid
flowchart TD
    S47["Stage 47\nWrong/random color-token decoded controls"]
    S48["Stage 48\nNeuron activation law:\nactivation distributions, top-k cumulative curves"]
    S49["Stage 49\nUnderstanding L62 heads knockout"]

    S47 --> C47["Test color-token specificity beyond object-token control"]
    S48 --> C48["Return to neuron-level regularities"]
    S49 --> C49["Test whether understanding head route can be blocked"]
```

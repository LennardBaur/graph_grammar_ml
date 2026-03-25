# DEG Paper: Detaillierte Schritt-fuer-Schritt Dokumentation

## Paper-Referenz
**"Data-Efficient Graph Grammar Learning for Molecular Generation"** (ICLR 2022, Oral)
Autoren: Minghao Guo, Veronika Thost, Beichen Li, Payel Das, Jie Chen, Wojciech Matusik

---

## Inhaltsverzeichnis

1. [Gesamtueberblick](#1-gesamtueberblick)
2. [Phase 1: Datenverarbeitung](#2-phase-1-datenverarbeitung)
   - 2.1 SMILES einlesen
   - 2.2 Molekuel → Hypergraph
   - 2.3 Cluster / Subgraphen finden
   - 2.4 GNN-Features extrahieren
   - 2.5 SubGraphSet aufbauen
3. [Phase 2: Eine MCMC-Iteration](#3-phase-2-eine-mcmc-iteration)
   - 3.1 Feature-Vektor zusammenbauen (302-dim)
   - 3.2 Agent-Sampling (Bernoulli)
   - 3.3 Ueberlappende Subgraphen mergen
   - 3.4 Produktionsregel erzeugen
   - 3.5 Hypergraph aktualisieren
4. [Phase 3: Kompletter MCMC-Durchlauf](#4-phase-3-kompletter-mcmc-durchlauf)
5. [Phase 4: Warum 5 MCMC Samples?](#5-phase-4-warum-5-mcmc-samples)
6. [Phase 5: REINFORCE Training](#6-phase-5-reinforce-training)
   - 6.1 Reward-Berechnung
   - 6.2 Policy Loss
   - 6.3 Backpropagation
7. [Phase 6: Molekuel-Generierung](#7-phase-6-molekuel-generierung)
8. [Glossar / Symboltabelle](#8-glossar)

---

## 1. Gesamtueberblick

**Paper:** Figure 1 (S.2) zeigt den Gesamtablauf:

```
Training Data (20 Molekuele)
    |
    v
Bottom-up Search (MCMC, gesteuert durch Agent F_theta)
    |
    v
Graph Grammar (Produktionsregeln)
    |
    v
Grammar Production (Regelanwendung)
    |
    v
Generated Samples
    |
    v
Evaluation (Diversity + Retro*) ---> Reward ---> REINFORCE ---> Update F_theta
```

**Code-Einstiegspunkt:** `main.py`, Zeilen 97-174 (`learn()`)

```
Epoche 1..50:
  fuer jedes der 5 MCMC-Samples:
    1. Kopiere Eingabedaten
    2. MCMC_sampling() -> fertige Grammatik
    3. random_produce() x100 -> generierte Molekuele
    4. Evaluate -> diversity + syn_rate -> Reward R
  Policy Loss berechnen (REINFORCE)
  Backprop + Optimizer Step
```

---

## 2. Phase 1: Datenverarbeitung

> **Paper:** Section 3 "Preliminaries" (S.3-4)
> **Code:** `grammar_generation.py`, Zeilen 13-54 (`data_processing()`)

### 2.1 SMILES einlesen

**Was passiert:** Rohe SMILES-Strings werden gelesen.

**Code:** `main.py`, Zeilen 188-191
```python
with open(args.training_data) as f:
    smiles_list = [line.strip() for line in f.readlines()]
```

**Output-Beispiel:**
```
[Molekuel 0]
  SMILES:     O=C=NCCCCCCN=C=O
  #Atome:     12
  #Bindungen: 11
  #Ringe:     0

[Molekuel 1]
  SMILES:     CC1=C(C=C(C=C1)CN=C=O)N=C=O
  #Atome:     14
  #Bindungen: 14
  #Ringe:     1
    Ring 0: Atome [1, 6, 5, 4, 3, 2] = ['C', 'C', 'C', 'C', 'C', 'C']
```

HDI ist eine lineare Kette (kein Ring), TDI hat einen Benzolring.

---

### 2.2 Molekuel → Hypergraph

> **Paper:** Section 3, S.3: "Given a molecule M, we derive a hypergraph H_M = (V, E_H)"
> **Paper:** Figure 2a (S.4): Zeigt die Hypergraph-Darstellung von Naphthalin-Diisocyanat

**Kernidee:** Ein Molekuel wird als Hypergraph dargestellt, wobei:
- **Hyperedges** = Atome (oder ganze Ringe als eine Hyperedge)
- **Knoten** = Bindungen zwischen Atomen

**Code:** `private/hypergraph.py`, Zeilen 722-759 (`mol_to_hg()`)
```python
def mol_to_hg(mol, kekulize, add_Hs):
    bipartite_g = mol_to_bipartite(mol, kekulize)  # Zeile 747
    hg = Hypergraph()                               # Zeile 748
    for each_atom in [...]:                          # Zeile 749
        node_set = set([])
        for each_bond in bipartite_g.adj[each_atom]:
            hg.add_node(each_bond, ...)              # Knoten = Bindung
            node_set.add(each_bond)
        hg.add_edge(node_set, ...)                   # Hyperedge = Atom
    return hg
```

**Output-Beispiel (HDI = O=C=NCCCCCCN=C=O):**
```
Hypergraph [Mol 0]:
  #Knoten (=Bonds): 11  |  #Edges (=Atome/Hyperedges): 12
  Edges (= Atome/Gruppen):
    e0: O (T) -- Knoten: bond_0(bond_type=2)          <- Sauerstoff, eine Doppelbindung
    e1: C (T) -- Knoten: bond_0(bond_type=2), bond_1(bond_type=2)  <- Kohlenstoff, zwei Doppelbindungen
    e2: N (T) -- Knoten: bond_1(bond_type=2), bond_2(bond_type=1)  <- Stickstoff, Doppel+Einfach
    ...
```

**Erklaerung:** Jeder Eintrag "e0: O (T)" bedeutet:
- `e0` = Hyperedge-ID (= Atom-Index)
- `O` = Atomsymbol (Sauerstoff)
- `(T)` = Terminal (= echtes Atom, kein Platzhalter)
- `bond_0(bond_type=2)` = Diese Hyperedge ist mit bond_0 verbunden, einer Doppelbindung

Fuer TDI mit dem Benzolring werden die 6 Ring-Atome NICHT in eine Ring-Hyperedge zusammengefasst (das passiert erst im naechsten Schritt bei den Subgraphen).

---

### 2.3 Cluster / Subgraphen finden

> **Paper:** Section 3, S.3-4: "We use the BRICS decomposition or tree decomposition"
> **Code:** `grammar_generation.py`, Zeilen 20-40 (innerhalb `data_processing()`)

**Was passiert:**
1. RDKit's `GetSymmSSSR()` findet alle Ringe
2. Einzelne Bindungen werden als Atom-Paare extrahiert
3. Ringe werden als Multi-Atom-Cluster behandelt
4. Jeder Cluster = ein `SubGraph`-Objekt

**Code:** `private/molecule_graph.py`, Zeilen 100-135 (`InputGraph.__init__()`)

**Output-Beispiel (HDI, keine Ringe):**
```
Subgraphen von [Mol 0]: 11 Stueck
  Subgraph 0: Atom-Indices=[0, 1]   SMILES=O=[CH2:1]       <- O=C Bindung
  Subgraph 1: Atom-Indices=[1, 2]   SMILES=[CH2:1]=[NH:1]  <- C=N Bindung
  Subgraph 2: Atom-Indices=[2, 3]   SMILES=[CH3:1][NH2:1]  <- N-C Bindung
  Subgraph 3: Atom-Indices=[3, 4]   SMILES=[CH3:1][CH3:1]  <- C-C Bindung
  ...
```

**Output-Beispiel (TDI, mit Benzolring):**
```
Subgraphen von [Mol 1]: 9 Stueck
  Subgraph 0: Atom-Indices=[0, 1]                SMILES=C[CH3:1]               <- Methyl-Gruppe
  Subgraph 1: Atom-Indices=[2, 3]                SMILES=[CH3:1][NH2:1]         <- C-N Bindung
  ...
  Subgraph 8: Atom-Indices=[1, 2, 6, 7, 12, 13] SMILES=c1c[CH:1]=[CH:1]c=[CH:1]1  <- BENZOLRING als ein Subgraph!
```

**Wichtig:** Der Benzolring (6 Atome) wird als EIN Subgraph behandelt, nicht als 6 einzelne.
Dies ist der Unterschied zwischen Hypergraph und normalem Graph -- eine Hyperedge kann mehr als 2 Knoten verbinden!

---

### 2.4 GNN-Features extrahieren

> **Paper:** Section 4.2, S.6 + Appendix B, S.13: "pretrained GIN from Hu et al. (2019)"
> **Code:** `GCN/feature_extract.py` (feature_extractor Klasse)

**Was passiert:** Ein vortrainiertes Graph Isomorphism Network (GIN) erzeugt fuer jedes Atom
einen 300-dimensionalen Embedding-Vektor. Fuer einen Subgraphen mit N Atomen werden
die N Atom-Embeddings **gemittelt** zu einem einzigen 300-dim Vektor.

**Code:** `private/molecule_graph.py`, Zeilen 137-144 (`get_subg_feature_for_agent()`)
```python
def get_subg_feature_for_agent(self, subg):
    atom_features = self.gnn_features  # 300-dim pro Atom
    subg_atoms = subg.subfrags         # Atom-Indices des Subgraphen
    return atom_features[subg_atoms].mean(axis=0)  # Mittelwert -> 300-dim
```

**Das GIN wird NICHT trainiert** -- es ist ein eingefrorener Feature-Extraktor.
Geladen aus: `GCN/model_gin/supervised_contextpred.pth`

---

### 2.5 SubGraphSet aufbauen

> **Paper:** Section 4.2, S.6: Die zwei Zusatz-Features (Haeufigkeit + Abdeckung)
> **Code:** `private/subgraph_set.py`, Zeilen 3-36 (`SubGraphSet`)

**Was passiert:** Das SubGraphSet trackt fuer jeden einzigartigen Subgraphen:
1. **Haeufigkeit:** Wie oft kommt er insgesamt vor?
2. **Abdeckung:** In wie vielen verschiedenen Molekuelen kommt er vor?

**Code:** `private/subgraph_set.py`, Zeilen 10-36 (`get_map_to_input()`)
```python
# map_to_input[MolKey(subg)] = [{input_mol_key: [(idx, subg), ...]}, count]
```

**Output-Beispiel:**
```
SubGraphSet: 6 unique Subgraphen
  'O=[CH2:1]':                     kommt 4x vor, in 2 Molekuel(en)  <- In BEIDEN Mol.
  '[CH2:1]=[NH:1]':                kommt 4x vor, in 2 Molekuel(en)  <- N=C=O Motiv
  '[CH3:1][NH2:1]':                kommt 4x vor, in 2 Molekuel(en)
  '[CH3:1][CH3:1]':                kommt 6x vor, in 2 Molekuel(en)  <- C-C am haeufigsten
  'C[CH3:1]':                      kommt 1x vor, in 1 Molekuel(en)  <- Nur in TDI (Methyl)
  'c1c[CH:1]=[CH:1]c=[CH:1]1':    kommt 1x vor, in 1 Molekuel(en)  <- Nur in TDI (Ring)
```

**Warum wichtig?** Diese Zahlen werden spaeter als 2 Zusatz-Features verwendet:
- `freq = 1 - exp(-count)` (z.B. 0.998 fuer count=6)
- `coverage = n_inputs / total_inputs` (z.B. 2/2 = 1.0)

Ein Subgraph der in ALLEN Molekuelen vorkommt, wird tendenziell eher selektiert
(da er eine allgemeinere Produktionsregel liefert).

---

## 3. Phase 2: Eine MCMC-Iteration

> **Paper:** Section 4.1, S.5, Figure 3
> **Code:** `grammar_generation.py`, Zeilen 57-117 (`grammar_generation()`)

Eine Iteration = ein Durchlauf ueber ALLE Molekuele, bei dem:
1. Der Agent fuer jeden Subgraphen entscheidet: selektieren (1) oder ignorieren (0)
2. Selektierte Subgraphen werden kontrahiert -> Produktionsregeln entstehen
3. Der Hypergraph schrumpft

### 3.1 Feature-Vektor zusammenbauen (302-dim)

> **Paper:** Section 4.2, S.6, Eq. (2)
> **Code:** `grammar_generation.py`, Zeilen 67-80

**Fuer jeden Subgraph wird ein 302-dimensionaler Feature-Vektor gebaut:**

```python
# grammar_generation.py, Zeilen 67-80
subg_feature = input_g.get_subg_feature_for_agent(subgraph)   # 300-dim GNN
num_occurance = subgraph_set.map_to_input[MolKey(subgraph)][1] # z.B. 6
num_in_input = len(subgraph_set.map_to_input[MolKey(subgraph)][0].keys())  # z.B. 2

final_feature = []
final_feature.extend(subg_feature.tolist())          # [0..299]: 300-dim GNN
final_feature.append(1 - np.exp(-num_occurance))     # [300]: Haeufigkeit (0..1)
final_feature.append(num_in_input / total_inputs)    # [301]: Abdeckung (0..1)
```

**Output-Beispiel:**
```
Subgraph '[CH3:1][CH3:1]': GNN-Feature(300-dim) + freq=0.998 + coverage=2/2
Subgraph 'C[CH3:1]':       GNN-Feature(300-dim) + freq=0.632 + coverage=1/2
```

C-C Bindung (freq=0.998) kommt viel haeufiger vor als die Methylgruppe (freq=0.632).

---

### 3.2 Agent-Sampling (Bernoulli)

> **Paper:** Section 4.2, S.6, Eq. (2):
> `X ~ Bernoulli(phi(e; theta))`,  `phi(e; theta) = sigma(-F_theta(f(e)))`
> **Code:** `agent.py`, Zeilen 7-35

**Architektur des Agent:**
```
302-dim Input -> Linear(302, 128) -> ReLU -> Dropout(0.5) -> Linear(128, 2) -> Softmax
```

**Code:** `agent.py`, Zeilen 7-19
```python
class Agent(nn.Module):
    def __init__(self, feat_dim, hidden_size):
        self.affine1 = nn.Linear(feat_dim + 2, hidden_size)  # 302 -> 128
        self.dropout = nn.Dropout(p=0.5)
        self.affine2 = nn.Linear(hidden_size, 2)             # 128 -> 2
    def forward(self, x):
        x = self.affine1(x)
        x = self.dropout(F.relu(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=-1)
```

**Sampling:** `agent.py`, Zeilen 22-35 (`sample()`)
```python
def sample(agent, features, iter_num, sample_number):
    probs = agent(features)              # MLP Forward -> [N, 2] Wahrscheinlichkeiten
    m = Categorical(probs)               # Kategorische Verteilung
    actions = m.sample()                  # Sample fuer jeden Subgraph: 0 oder 1
    log_probs = m.log_prob(actions)       # Log-Prob fuer REINFORCE speichern
    agent.saved_log_probs[sample_number][iter_num] = log_probs
    return actions.numpy(), any(actions == 1)
```

**Output-Beispiel (Mol 0, HDI):**
```
Agent-Entscheidung (0=ignorieren, 1=selektieren):
  Subgraph 0: action=1  >>> SELEKTIERT <<<  Indices=[0, 1]  SMILES=O=[CH2:1]
  Subgraph 1: action=1  >>> SELEKTIERT <<<  Indices=[1, 2]  SMILES=[CH2:1]=[NH:1]
  Subgraph 2: action=0      ignoriert       Indices=[2, 3]  SMILES=[CH3:1][NH2:1]
  Subgraph 3: action=0      ignoriert       Indices=[3, 4]  SMILES=[CH3:1][CH3:1]
  ...
```

Die Entscheidung ist **stochastisch** -- jedes Mal andere Selektionen!
Das ist der Kern des MCMC-Samplings.

---

### 3.3 Ueberlappende Subgraphen mergen

> **Paper:** Section 4.1, S.5: "Extract all connected components"
> **Code:** `private/molecule_graph.py`, Zeilen 193-260 (`merge_selected_subgraphs()`)

**Was passiert:** Selektierte Subgraphen die Atome teilen werden zusammengefuegt.

**Beispiel aus dem Output:**
```
Selektiert: Subgraph 0 [0,1] + Subgraph 1 [1,2]  (teilen Atom 1!)
  -> Merged Subgraph 0: SMILES=O=C=[NH:1]  Indices=[0, 1, 2]

Selektiert: Subgraph 5 [5,6] + Subgraph 6 [6,7] + Subgraph 7 [8,7]  (Kette!)
  -> Merged Subgraph 1: SMILES=C(C[CH3:1])[CH3:1]  Indices=[5, 6, 7, 8]

Selektiert: Subgraph 9 [9,10]  (allein, kein Overlap)
  -> Merged Subgraph 2: SMILES=[CH2:1]=[NH:1]  Indices=[9, 10]
```

**Visuell fuer HDI (O=C=N-C-C-C-C-C-C-N=C=O):**
```
Atome:  0  1  2  3  4  5  6  7  8  9  10  11
        O= C= N- C- C- C- C- C- C- N= C=  O
        |------|        |-----------|  |---|
        Merged 0        Merged 1       Merged 2
        (O=C=N)         (CCCC)         (N=C)
```

---

### 3.4 Produktionsregel erzeugen

> **Paper:** Section 4.1, S.5, Eq. (1)
> **Code:** `private/grammar.py`, Zeilen 835-992 (`generate_rule()`)

**Was passiert fuer jeden gemergten Subgraph:**

1. **RHS (Right-Hand-Side) bauen** (Zeilen 891-953):
   - Kopiere alle Atome/Bindungen des Subgraphen
   - Finde **Anker-Knoten**: Atome die den Subgraph mit dem Rest verbinden
   - Diese Anker werden als "externe Knoten" mit `ext_id` markiert

2. **LHS (Left-Hand-Side) bauen** (Zeilen 959-983):
   - Ein einzelner Non-Terminal (NT) Knoten `R*`
   - Verbunden mit den Anker-Knoten

3. **Regel = LHS -> RHS**

**Paper Eq. (1):**
```
LHS := H(V_L, E_L), V_L = {R*} union V_anc
RHS := H(V_R, E_R), V_R = V_sub union V_anc
```

**Beispiel:** Fuer Merged Subgraph 0 (O=C=N, Indices [0,1,2]):
- Anker: Atom 3 (C), da Atom 2 (N) mit Atom 3 verbunden ist
- LHS: NT mit degree=1 (ein Anker)
- RHS: O, C, N (3 Atome) + 1 Anker-Verbindung

**Output-Beispiel:**
```
-> NEUE Regel 0 [END] erzeugt!     <- O=C=N Fragment (degree=1, 3 Atome)
-> NEUE Regel 1 [END] erzeugt!     <- C-C-C-C Fragment (degree=2, 4 Atome)
-> NEUE Regel 2 [END] erzeugt!     <- N=C Fragment (degree=2, 2 Atome)
```

**Regel-Typen:**
- **END:** RHS enthaelt NUR Terminal-Atome (keine NT-Knoten) -> Stoppt die Generierung
- **EXPANSION:** RHS enthaelt Terminal-Atome UND NT-Knoten -> Kann weiter expandiert werden
- **START:** LHS ist das initiale Symbol X -> Erster Schritt der Generierung

In der ersten Iteration entstehen nur END-Regeln, da noch nichts vorher kontrahiert wurde.

**Code:** `private/grammar.py`, Zeilen 684-727 (`append()`)
```python
def append(self, prod_rule):
    for each_idx, each_prod_rule in enumerate(self.prod_rule_list):
        is_same, isomap = prod_rule.is_same(each_prod_rule, ignore_order=True)
        if is_same:
            return each_idx, prod_rule  # Duplikat!
    self.prod_rule_list.append(prod_rule)  # Neue Regel
    return len(self.prod_rule_list)-1, prod_rule
```

**Output bei TDI (Mol 1):**
```
-> NEUE Regel 3 [END] erzeugt!     <- Methylgruppe (C-CH3)
-> NEUE Regel 4 [END] erzeugt!     <- C-N Bindung (fuer Isocyanat-Seitenkette)
-> NEUE Regel 5 [END] erzeugt!     <- O=C (Isocyanat-Ende)
-> Regel existiert bereits (Duplikat)  <- O=C Fragment war schon da!
-> NEUE Regel 6 [END] erzeugt!     <- C-C-N Fragment
```

---

### 3.5 Hypergraph aktualisieren

> **Paper:** Section 4.1, S.5: "Update H_{M,t+1} by replacing each component with R*"
> **Code:** `private/molecule_graph.py`, Zeilen 262-296 (`update_subgraph()`)

**Was passiert:** Die kontrahierten Teile werden als "besucht" markiert und in den
verbleibenden Subgraphen absorbiert. Die verbleibenden Subgraphen wachsen dadurch.

**Output-Beispiel (HDI nach 1 Iteration):**
```
VORHER: 11 Subgraphen (jeder = 1 Bindung)
NACHHER: 5 Subgraphen (groessere Fragmente)
  Subgraph 0: Indices=[0,1,2,3]     SMILES=O=C=N[CH3:1]      <- Gewachsen!
  Subgraph 1: Indices=[3,4]         SMILES=[CH3:1][CH3:1]     <- Unveraendert
  Subgraph 2: Indices=[4,5,6,7,8]   SMILES=C(C[CH3:1])C[CH3:1]
  Subgraph 3: Indices=[5,6,7,8,9,10] SMILES=C(CN=[CH2:1])C[CH3:1]
  Subgraph 4: Indices=[9,10,11]      SMILES=O=C=[NH:1]
```

---

## 4. Phase 3: Kompletter MCMC-Durchlauf

> **Paper:** Section 4.1, S.5
> **Code:** `grammar_generation.py`, Zeilen 120-133 (`MCMC_sampling()`)

**Was passiert:** Phase 2 wird WIEDERHOLT bis alle Subgraphen kontrahiert sind.

```python
# grammar_generation.py, Zeilen 120-133
def MCMC_sampling(agent, input_graphs_dict, subgraph_set, grammar, sample_number, args):
    iter_num = 0
    while True:
        done_flag, new_igd, new_ss, new_g = grammar_generation(
            agent, input_graphs_dict, subgraph_set, grammar, iter_num, sample_number, args
        )
        if done_flag:
            break
        input_graphs_dict = deepcopy(new_igd)
        subgraph_set = deepcopy(new_ss)
        grammar = deepcopy(new_g)
        iter_num += 1
    return iter_num, grammar, input_graphs_dict
```

**Output-Beispiel (6 Iterationen bis fertig):**
```
MCMC Iteration 0: 20 Subgraphen -> 9 Subgraphen    (5 neue Regeln, alle END)
MCMC Iteration 1:  9 Subgraphen -> 7 Subgraphen    (7 Regeln, erste EXPANSION!)
MCMC Iteration 2:  7 Subgraphen -> 5 Subgraphen    (9 Regeln)
MCMC Iteration 3:  5 Subgraphen -> 2 Subgraphen    (11 Regeln)
MCMC Iteration 4:  2 Subgraphen -> 0 Subgraphen    (13 Regeln, START-Regeln!)
MCMC Iteration 5:  0 -> FERTIG!
```

**Warum entstehen verschiedene Regeltypen zu verschiedenen Zeitpunkten?**

| Iteration | Was passiert | Regeltyp |
|-----------|-------------|----------|
| Frueh (0-1) | Kleine Fragmente kontrahiert (O=C, C-C, N-C) | **END** (nur Atome, keine NT) |
| Mitte (2-3) | Groessere Fragmente, enthalten bereits kontrahierte Teile | **EXPANSION** (Atome + NT-Knoten) |
| Ende (4) | Gesamtes Molekuel wird zu einem Knoten | **START** (LHS = X) |

**Fertige Grammatik (13 Regeln):**
```
Regel  0 [END]:       NT(degree=2) -> 3 Atome: [N, C, C]
Regel  1 [END]:       NT(degree=2) -> 2 Atome: [C, C]
Regel  2 [END]:       NT(degree=1) -> 2 Atome: [C, O]
...
Regel  5 [EXPANSION]: NT(degree=2) -> 0 Atome, 2 NT-Edges   <- Verbindet 2 Teilbaeume
...
Regel 11 [START]:     X -> 0 Atome, 2 NT-Edges              <- HDI: Zwei Enden
Regel 12 [START]:     X -> 3 Atome [C,C,C], 3 NT-Edges      <- TDI: Ring + 3 Seitenketten
```

**Warum 2 START-Regeln?** Weil wir 2 verschiedene Molekuele haben:
- HDI (linear): START -> 2 NT-Knoten (linke und rechte Haelfte)
- TDI (Ring): START -> 3 Atome (Ring-Kern) + 3 NT (Methyl + 2 Isocyanat-Gruppen)

---

## 5. Phase 4: Warum 5 MCMC Samples?

> **Paper:** Section 4.2, S.6 + Appendix B, S.13: "MC sampling size N = 5"
> **Code:** `main.py`, Zeilen 123-145 (innere Schleife ueber `MCMC_size`)

**Kernproblem:** Das Agent-Sampling ist STOCHASTISCH. Jeder Durchlauf erzeugt
eine ANDERE Grammatik, weil andere Subgraphen selektiert werden.

**Warum 5 und nicht 1?**
1. **Exploration:** Verschiedene Grammatiken erkunden verschiedene "Zerlegungsstrategien"
2. **Varianzreduktion:** Mittelung ueber 5 Returns reduziert die Varianz des Gradientschaetzers
3. **Besseres Lernsignal:** Der Agent sieht: "Strategie A gab Reward 1.5, Strategie B nur 0.8"

**Output-Beispiel (3 Samples statt 5 fuer Geschwindigkeit):**
```
Sample   #Regeln    #Generiert   Diversity
0        12         5            0.7983
1        10         5            0.7917
2        11         5            0.7373
```

**Beobachtung:** Gleiche 2 Eingabe-Molekuele, aber:
- Sample 0: 12 Regeln, Diversity 0.80
- Sample 2: 11 Regeln, Diversity 0.74

Die unterschiedlichen Selektions-Entscheidungen fuehren zu unterschiedlich
ausdruecksstarken Grammatiken!

---

## 6. Phase 5: REINFORCE Training

> **Paper:** Section 4.2, S.6, Eq. (3) + (4)
> **Code:** `main.py`, Zeilen 148-162

### 6.1 Reward-Berechnung

> **Paper:** Section 5.1, S.7 + Appendix B, S.13

```
R = lambda_1 * diversity + lambda_2 * syn_rate
  = 1 * diversity   + 2 * syn_rate
```

- **diversity:** Mittlere paarweise Tanimoto-Distanz der generierten Molekuele
  (Code: `private/metrics.py`, InternalDiversity Klasse)
- **syn_rate:** Anteil der Molekuele fuer die Retro* einen Syntheseweg findet
  (Code: `main.py`, Zeilen 65-94, `retro_sender()`)

**Output-Beispiel:**
```
Sample 0: Diversity=0.7312, Syn(dummy)=0.3675, R = 0.7312 + 2*0.3675 = 1.4661
Sample 1: Diversity=0.7760, Syn(dummy)=0.1903, R = 0.7760 + 2*0.1903 = 1.1567
Sample 2: Diversity=0.7047, Syn(dummy)=0.3780, R = 0.7047 + 2*0.3780 = 1.4607
```

### 6.2 Policy Loss

> **Paper:** Eq. (4), S.6

**Schritt 1: Returns normalisieren (Baseline-Subtraktion)**
```
Rohe Returns:         [1.4661, 1.1567, 1.4607]
Mittelwert:           1.3612
Normalisiert:         [0.1050, -0.2045, 0.0995]
```

Sample 1 hat den niedrigsten Reward -> negativer normalisierter Return.
Der Agent wird bestraft fuer die Entscheidungen in Sample 1.

**Schritt 2: Policy Loss berechnen**

> Code: `main.py`, Zeilen 148-157

```python
# main.py, Zeilen 148-157
policy_loss = torch.tensor([0.])
for sample_number in agent.saved_log_probs.keys():
    max_iter_num = max(list(agent.saved_log_probs[sample_number].keys()))
    for iter_num in agent.saved_log_probs[sample_number].keys():
        log_probs = agent.saved_log_probs[sample_number][iter_num]
        for log_prob in log_probs:
            policy_loss += (-log_prob * gamma ** (max_iter_num - iter_num)
                           * returns[sample_number]).sum()
```

**Formel:** `Loss = -sum( log_prob * gamma^(T-t) * R_normalized )`

- `log_prob`: Log-Wahrscheinlichkeit der Agent-Entscheidung
- `gamma^(T-t)`: Discount-Faktor (spaetere Entscheidungen zaehlen mehr, da T-t kleiner)
- `R_normalized`: Normalisierter Return

**Output:**
```
Gesamte Entscheidungen: 18
Policy Loss: -0.328252
```

### 6.3 Backpropagation

**Code:** `main.py`, Zeilen 160-162
```python
optimizer.zero_grad()
policy_loss.backward()
optimizer.step()
agent.saved_log_probs.clear()
```

**Output:**
```
Gewichte vorher (erste 3): [-0.0221, 0.0396, -0.0095]
Gewichte nachher (erste 3): [-0.0121, 0.0496, 0.0005]
Differenz:                  [+0.0100, +0.0100, +0.0100]
```

Die Gewichte aendern sich! Nach 50 Epochen mit je 5 Samples wird der Agent
bessere Selektions-Entscheidungen treffen.

---

## 7. Phase 6: Molekuel-Generierung

> **Paper:** Appendix A, S.13 + Section 3, S.4, Figure 2c
> **Code:** `grammar_generation.py`, Zeilen 136-189 (`random_produce()`)

### Generierungs-Algorithmus:

```python
# grammar_generation.py, Zeilen 136-189
def random_produce(grammar):
    # 1. Starte mit leerem Hypergraph
    hg = Hypergraph()

    # 2. Waehle eine START-Regel und wende sie an
    start_rules = grammar.start_rule_list
    start_rule = random.choice(start_rules)
    hg = start_rule.graph_rule_applied_to(hg)

    # 3. Iterativ: finde NT-Knoten, waehle passende Regel
    for iter in range(max_iterations):
        nt_edges = hg.get_all_NT_edges()
        if len(nt_edges) == 0:
            break  # Keine NT mehr -> fertig!

        nt_edge = random.choice(nt_edges)
        matching_rules = grammar.get_prod_rules_with_lhs(nt_edge)

        # Regel-Auswahl mit Terminierungs-Bias:
        probs = [exp(0.5 * iter * is_ending(rule)) for rule in matching_rules]
        probs = probs / sum(probs)
        rule = random.choice(matching_rules, p=probs)

        hg = rule.graph_rule_applied_to(hg)

    # 4. Konvertiere Hypergraph -> Molekuel
    mol = hg_to_mol(hg)
    return mol
```

**Terminierungs-Bias (Paper Appendix A):**
```
p(r) = Z^{-1} * exp(alpha * t * x_r)
```
- `alpha = 0.5`
- `t` = aktuelle Iteration
- `x_r = 1` wenn Regel nur Terminal-Atome hat (END-Regel), sonst 0

Bei Iteration 0: END und EXPANSION gleich wahrscheinlich.
Bei Iteration 10: END-Regeln sind `exp(0.5 * 10) = 148x` wahrscheinlicher!

**Output-Beispiel:**
```
Molekuel 1:  O=C=O                             (in 2 Iterationen)  <- Sehr kurz
Molekuel 5:  Cc1ccc(CN=C=O)cc1N=C=O            (in 7 Iterationen)  <- = TDI (Trainingsmolekuel!)
Molekuel 7:  O=C=NCCCCCCCCCCN=C=O              (in 7 Iterationen)  <- HDI-aehnlich, aber laenger
Molekuel 9:  O=C=NCCCCCCCCCCCCCN=C=O           (in 10 Iterationen) <- Noch laenger
Molekuel 10: O=C=C=NC1=NCCCCCC=CC(CN=O)=C1     (in 7 Iterationen)  <- Komplett NEU!
```

**Beobachtungen:**
1. Manche Molekuele sind Trainingsmolekulen aehnlich (Interpolation)
2. Manche sind komplett neu (Extrapolation)
3. Alle behalten die Isocyanat-Grundstruktur (N=C=O Gruppen)
4. Diversity = 0.81 (gut!)

---

## 8. Glossar

| Begriff | Paper | Code | Erklaerung |
|---------|-------|------|------------|
| **Hypergraph** | Sec. 3, S.3 | `private/hypergraph.py` | Graph wo Edges mehr als 2 Knoten verbinden koennen (fuer Ringe) |
| **Hyperedge** | Sec. 3, S.3 | Edge in `Hypergraph` | Repr. ein Atom oder einen Ring |
| **Terminal (T)** | Sec. 3, S.4 | `TSymbol` in `symbol.py` | Echtes Atom (O, C, N, ...) |
| **Non-Terminal (NT)** | Sec. 3, S.4 | `NTSymbol` in `symbol.py` | Platzhalter `R*`, wird weiter expandiert |
| **Produktionsregel** | Sec. 3, S.4 | `ProductionRule` in `grammar.py:30` | LHS -> RHS (z.B. R* -> O-C-N) |
| **START-Regel** | Sec. 4.1, S.5 | `is_start_rule` in `grammar.py:46` | LHS = X (Anfangssymbol) |
| **END-Regel** | Sec. 4.1, S.5 | `is_ending` in `grammar.py:56` | RHS hat keine NT -> Stoppt |
| **EXPANSION-Regel** | Sec. 4.1, S.5 | (nicht END, nicht START) | RHS hat T + NT -> Geht weiter |
| **Anker-Knoten** | Eq. (1), S.5 | `ext_id` in Hypergraph | Verbindung zwischen Fragment und Rest |
| **F_theta** | Sec. 4.2, S.6 | `Agent` in `agent.py:7` | MLP (302->128->2), gewichtet Hyperedges |
| **GIN** | App. B, S.13 | `GCN/feature_extract.py` | Pretrained GNN, erzeugt 300-dim Embeddings |
| **MCMC Sample** | Sec. 4.2, S.6 | `MCMC_sampling()` | Ein kompletter Grammatik-Konstruktions-Durchlauf |
| **Diversity** | Sec. 5.1, S.7 | `InternalDiversity` in `metrics.py` | Mittlere paarweise Tanimoto-Distanz |
| **syn_rate** | Sec. 5.1, S.7 | `retro_sender()` in `main.py:65` | Retro*-Synthesizability |
| **REINFORCE** | Eq. (4), S.6 | `main.py:148-157` | Policy Gradient zum Optimieren des Agents |
| **gamma** | Impl. Detail | `main.py:149` | Discount-Faktor fuer zeitliche Gewichtung |
| **SubGraphSet** | Impl. Detail | `subgraph_set.py:3` | Trackt unique Subgraphen ueber alle Molekuele |

---

## Datei-Uebersicht

| Datei | Zeilen | Rolle | Paper-Abschnitt |
|-------|--------|-------|-----------------|
| `main.py` | 97-174 | Trainingsschleife (50 Epochen, 5 Samples) | Sec. 4.2, Eq. 3+4 |
| `grammar_generation.py` | 13-189 | data_processing, grammar_generation, MCMC_sampling, random_produce | Sec. 3, 4.1, App. A |
| `agent.py` | 7-35 | Agent MLP + Bernoulli-Sampling | Sec. 4.2, Eq. 2 |
| `private/grammar.py` | 30-992 | ProductionRule, ProductionRuleCorpus, generate_rule | Sec. 3, Eq. 1 |
| `private/hypergraph.py` | 15-828 | Hypergraph, mol_to_hg, hg_to_mol | Sec. 3, Fig. 2a |
| `private/molecule_graph.py` | 9-296 | MolGraph, InputGraph, SubGraph, merge | Sec. 4.1, Fig. 3 |
| `private/subgraph_set.py` | 3-51 | SubGraphSet (cross-molecule tracking) | Sec. 4.2 (Features) |
| `private/symbol.py` | 1-176 | TSymbol, NTSymbol, BondSymbol | Sec. 3 |
| `private/metrics.py` | - | InternalDiversity, Retro* Wrapper | Sec. 5.1 |
| `GCN/feature_extract.py` | - | Pretrained GIN Feature-Extraktor | App. B, App. D |

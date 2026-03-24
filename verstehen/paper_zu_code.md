# DEG: Paper-zu-Code Dokumentation

> **Paper:** "Data-Efficient Graph Grammar Learning for Molecular Generation" (ICLR 2022)
> **Autoren:** Minghao Guo, Veronika Thost, Beichen Li, Payel Das, Jie Chen, Wojciech Matusik

---

## Inhaltsverzeichnis

1. [Gesamtueberblick](#1-gesamtueberblick)
2. [Arbeitspaket 1: Datenverarbeitung & Hypergraph-Repraesentation](#arbeitspaket-1)
3. [Arbeitspaket 2: Bottom-Up Grammatik-Konstruktion](#arbeitspaket-2)
4. [Arbeitspaket 3: Agent (Potential Function) & Feature-Extraktion](#arbeitspaket-3)
5. [Arbeitspaket 4: REINFORCE Training Loop](#arbeitspaket-4)
6. [Arbeitspaket 5: Grammatik-Produktion (Molekuel-Generierung)](#arbeitspaket-5)
7. [Arbeitspaket 6: Evaluation & Metriken](#arbeitspaket-6)
8. [Arbeitspaket 7: Retro-Star Synthesizability](#arbeitspaket-7)

---

## 1. Gesamtueberblick

### Was macht das System?

Das System lernt eine **Graph-Grammatik** aus wenigen Molekuelen (~20) und kann damit neue, valide und synthetisierbare Molekuele generieren.

**Paper Sec. 1, S.1 + Figure 1 (S.2):**
Der Ablauf ist:
1. **Training Data** (SMILES) -> Molekuele als Hypergraphen darstellen
2. **Bottom-Up Search** -> Hyperedges iterativ kontrahieren, dabei Produktionsregeln extrahieren
3. **Graph Grammar** -> Sammlung von Produktionsregeln (LHS -> RHS)
4. **Grammar Production** -> Regeln anwenden um neue Molekuele zu erzeugen
5. **Evaluation** -> Diversity + Synthesizability messen
6. **Update Edge Weight Function F_theta** -> Agent (MLP) per REINFORCE optimieren

### Hauptdateien und ihre Rollen

| Datei | Rolle | Paper-Bezug |
|-------|-------|-------------|
| `main.py` | Training-Loop (50 Epochen, 5 MCMC Samples, Policy Gradient) | Sec. 4.2, Algorithmus |
| `grammar_generation.py` | Datenverarbeitung, Grammatik-Konstruktion, MCMC, Produktion | Sec. 4.1 + 4.2 |
| `agent.py` | Agent MLP (F_theta) + Sampling | Sec. 4.2 (Eq. 2) |
| `private/grammar.py` | ProductionRule, ProductionRuleCorpus, generate_rule() | Sec. 3 + 4.1 (Eq. 1) |
| `private/hypergraph.py` | Hypergraph-Klasse, mol_to_hg(), hg_to_mol() | Sec. 3 (Molecular Hypergraph) |
| `private/molecule_graph.py` | MolGraph, SubGraph, InputGraph, MolKey | Sec. 4.1 (Datenstrukturen) |
| `private/subgraph_set.py` | SubGraphSet (molekueluebergreifendes Subgraph-Tracking) | Sec. 4.1 (simultane Konstruktion) |
| `private/metrics.py` | InternalDiversity (Tanimoto) | Sec. 5.1 |
| `GCN/feature_extract.py` | Pretrained GIN Feature Extractor f(e) | Sec. 4.2 + Appendix B |
| `retro_star_listener.py` | Retro* Worker-Prozess (File-IPC) | Sec. 5.1 (RS Metrik) |

---

## Arbeitspaket 1: Datenverarbeitung & Hypergraph-Repraesentation <a name="arbeitspaket-1"></a>

### Paper-Kontext

**Paper Sec. 3, "Molecular Hypergraph" (S.3):**
> Molekuele werden als Hypergraphen dargestellt: H_M = (V, E_H).
> - Jeder Bond der nur 2 Atome verbindet -> eine Hyperedge
> - Jeder Ring (inkl. aromatisch) -> eine Hyperedge die alle Atome des Rings verbindet
> Siehe Figure 2(a) fuer ein Beispiel (Naphthalin-Diisocyanat).

### Code-Ablauf

#### Schritt 1: SMILES einlesen
**Datei:** `main.py:197-202`
```
Liest die Trainingsdaten aus einer Textdatei (z.B. datasets/isocyanates.txt).
Entfernt Duplikate.
```

#### Schritt 2: data_processing()
**Datei:** `grammar_generation.py:13-54`

Fuer jedes Molekuel:
1. **Kekulisierung** (Z.24): `smiles = get_smiles(get_mol(smiles))` -- stellt sicher, dass aromatische Bindungen explizit als Einzel-/Doppelbindungen dargestellt werden.

2. **Cluster-Findung** (Z.28-36):
   - Ohne `--motif`: `find_clusters(mol)` aus `fuseprop` -- findet Ringe und einzelne Bindungen als Cluster
   - Mit `--motif`: `find_fragments(mol)` -- findet groessere Motive (fuer Polymere)
   - **Paper-Bezug:** Sec. 3 -- "a hyperedge is added for each bond that joins only two nodes, and for each individual ring"

3. **SubGraph-Objekte erstellen** (Z.38-43): Fuer jeden Cluster wird ein `SubGraph` erzeugt
   - `SubGraph` erbt von `MolGraph` (`private/molecule_graph.py:89-97`)
   - Speichert: `mol` (RDKit Mol), `mapping_to_input_mol`, `subfrags` (Atom-Indices)

4. **InputGraph erstellen** (Z.48): `InputGraph(mol, smiles, subgraphs, subgraphs_idx, GNN_model_path)`
   - **Datei:** `private/molecule_graph.py:100-116`
   - Erzeugt intern den Hypergraphen: `self.hypergraph = mol_to_hg(mol)` (Z.15 in MolGraph.__init__)

5. **SubGraphSet erstellen** (Z.53): Trackt identische Subgraphen ueber alle Molekuele hinweg
   - **Datei:** `private/subgraph_set.py:3-36`
   - `map_to_input`: Dict das fuer jeden unique Subgraph zaehlt: In welchen Molekuelen kommt er vor? Wie oft insgesamt?

#### Schritt 3: Molekuel -> Hypergraph Konvertierung
**Datei:** `private/hypergraph.py` -- Funktion `mol_to_hg()` (suche nach `def mol_to_hg`)

Die `Hypergraph`-Klasse (Z.15-36):
- Nutzt intern `nx.Graph` als bipartiten Graph
- `nodes`: Repraesentieren Bonds/Bindungen (Knotenattribut: `BondSymbol`)
- `edges`: Repraesentieren Atome/Hyperedges (Kantenattribut: `TSymbol` fuer terminal, `NTSymbol` fuer nicht-terminal)
- **ACHTUNG Namenskonvention:** Im Code sind "nodes" = Bonds und "edges" = Atome/Hyperedges (umgekehrt zur Intuition!)

---

## Arbeitspaket 2: Bottom-Up Grammatik-Konstruktion <a name="arbeitspaket-2"></a>

### Paper-Kontext

**Paper Sec. 4.1, "Bottom-up Grammar Construction" (S.5) + Figure 3 (S.5):**
> Ein Bottom-Up-Verfahren baut Produktionsregeln auf, indem es iterativ Hyperedges kontrahiert.
> In jeder Iteration t:
> 1. Betrachte den aktuellen Hypergraphen H_{M,t}
> 2. Sample eine Menge von Hyperedges E_t* (via Potential Function F_theta)
> 3. Extrahiere verbundene Komponenten
> 4. Konvertiere jede Komponente in eine Produktionsregel (Eq. 1)
> 5. Ersetze die Komponenten durch NT-Knoten -> neuer Hypergraph H_{M,t+1}

**Paper Eq. 1 (S.5):**
```
LHS := H(V_L, E_L), V_L = {R*} union V_anc, E_L = {(R*, v) | v in V_anc}
RHS := H(V_R, E_R), V_R = V_sub union V_anc, E_R = E_anc union E_sub
```
- V_anc = Anchor-Knoten (verbinden die Substruktur mit dem Rest des Molekuels)
- R* = Non-Terminal Knoten
- LHS hat einen NT-Knoten mit Anchor-Knoten
- RHS hat die vollstaendige Substruktur plus Anchor-Knoten

### Code-Ablauf

#### MCMC_sampling()
**Datei:** `grammar_generation.py:120-133`

```python
def MCMC_sampling(agent, all_input_graphs_dict, all_subgraph_set, all_grammar, sample_number, args):
    iter_num = 0
    while(True):
        done_flag, ... = grammar_generation(agent, ...)
        if done_flag:
            break
        iter_num += 1
    return iter_num, new_grammar, new_input_graphs_dict
```
- Ruft `grammar_generation()` wiederholt auf bis alle Hyperedges kontrahiert sind
- **Paper-Bezug:** "The above process continues until the hypergraph only consists of one single non-terminal node." (Sec. 4.1, S.5)

#### grammar_generation() -- Eine Iteration
**Datei:** `grammar_generation.py:57-117`

Fuer jedes Eingabe-Molekuel (Z.75):

1. **Feature-Berechnung** (Z.80-89): Fuer jeden noch vorhandenen Subgraphen:
   - GNN-Feature (300-dim) via `get_subg_feature_for_agent()` (molecule_graph.py:137-144)
   - Plus 2 Zusatzfeatures:
     - `1 - exp(-num_occurance)`: Wie oft kommt dieser Subgraph insgesamt vor?
     - `num_in_input / total_inputs`: In wie vielen Molekuelen kommt er vor?
   - -> 302-dimensionaler Feature-Vektor pro Subgraph
   - **Paper Sec. 4.2 (S.6):** "F(e;theta) = F_theta(f(e))" wobei f(e) = GNN Features

2. **Agent-Sampling** (Z.90-93): `sample(agent, features, iter_num, sample_number)`
   - Agent entscheidet fuer JEDEN Subgraph: selektieren (1) oder nicht (0)?
   - Bernoulli-Verteilung ueber die Ausgabe des MLP
   - **Paper Eq. 2 (S.6):** "X ~ Bernoulli(phi(e;theta)), phi(e;theta) = P(X=1) = sigma(-F_theta(f(e)))"
   - **ACHTUNG:** Im Code ist die Logik leicht anders -- der Agent gibt Softmax ueber [nicht-selektieren, selektieren] aus

3. **Merge selektierter Subgraphen** (Z.100): `merge_selected_subgraphs(action_list)`
   - **Datei:** `private/molecule_graph.py:193-260`
   - Verbindet ueberlappende selektierte Subgraphen zu groesseren Subgraphen
   - **Paper-Bezug:** Sec. 4.1 -- "extract all connected components with respect to these hyperedges"

4. **Regel-Generierung** (Z.102-110):
   - Fuer jeden zusammengefuehrten Subgraph: `generate_rule(input_g, subg, grammar)`
   - **Datei:** `private/grammar.py:835-992`
   - Dann `update_subgraph()` um den Hypergraphen zu aktualisieren

#### generate_rule() im Detail
**Datei:** `private/grammar.py:835-992`

**Paper Eq. 1 (S.5)** wird hier implementiert:

1. **Ext-Nodes finden** (Z.886-938):
   - Iteriert ueber alle Edges des Subgraphen
   - Fuer unbesuchte Edges: findet Knoten die ausserhalb des Subgraphen liegen -> ext_nodes (= V_anc)
   - Fuer besuchte Edges: nutzt "watershed" Mechanismus um vorherige NT-Knoten zu tracken

2. **RHS konstruieren** (Z.891-953):
   - Kopiert den Subgraphen mit allen Attributen
   - Fuegt ext_nodes hinzu
   - Markiert besuchte Bereiche als NT-Edges

3. **LHS konstruieren** (Z.960-983):
   - Wenn keine ext_nodes -> Startregel (LHS ist leer)
   - Sonst: LHS = eine NT-Hyperedge mit den ext_nodes
   - **Paper:** "For this finally constructed rule, we use the initial node X instead of R* on the left-hand side." (S.5)

4. **Zur Grammatik hinzufuegen** (Z.988-990):
   - `grammar.append(rule)` prueft ob die Regel schon existiert (via `is_same()`)
   - **Datei:** `private/grammar.py:527-626` -- `ProductionRuleCorpus`

#### Produktionsregel-Typen
**Datei:** `private/grammar.py:30-57`

| Typ | Bedingung | Paper-Bezug |
|-----|-----------|-------------|
| **Start-Regel** | `lhs.num_nodes == 0` (Z.47) | Figure 2(b): p1 (X -> ...) |
| **Expansion-Regel** | LHS hat NT, RHS hat weitere NTs | Figure 2(b): p3, p4 |
| **End-Regel** | `rhs` hat keine NT-Edges (Z.57) | Figure 2(b): p2 |

---

## Arbeitspaket 3: Agent (Potential Function) & Feature-Extraktion <a name="arbeitspaket-3"></a>

### Paper-Kontext

**Paper Sec. 4.2, "Optimizing Grammar Construction" (S.6):**
> "We define our edge weight function as F(e;theta) = F_theta(f(e))"
> - f(e) = Feature Extractor (pretrained GNN), theta = Parameter von F_theta
> - F_theta = "potential function" (neuronales Netz)
> - Ausgabe: phi(e;theta) = sigma(-F_theta(f(e)))
> - Groesseres F -> kleinere Wahrscheinlichkeit dass die Hyperedge selektiert wird

**Paper Appendix B (S.13):**
> "For the potential function F_theta, we use a two-layer fully connected network with size 300 and 128."
> Feature Extractor: pretrained GIN (Graph Isomorphism Network) mit 300-dim Node Embeddings.

### Code

#### Agent-Architektur
**Datei:** `agent.py:7-19`

```python
class Agent(nn.Module):
    def __init__(self, feat_dim, hidden_size):
        self.affine1 = nn.Linear(feat_dim + 2, hidden_size)  # 302 -> 128
        self.dropout = nn.Dropout(p=0.5)
        self.affine2 = nn.Linear(hidden_size, 2)             # 128 -> 2
```

- Input: 302 = 300 (GNN) + 2 (Haeufigkeits-Features)
- Hidden: 128
- Output: 2 (Softmax ueber [nicht-selektieren, selektieren])
- **Paper vs. Code Unterschied:** Paper beschreibt Bernoulli mit sigma(-F_theta), Code nutzt Categorical ueber Softmax. Effekt ist aequivalent.

#### Sample-Funktion
**Datei:** `agent.py:22-35`

```python
def sample(agent, subgraph_feature, iter_num, sample_number):
    prob = agent(subgraph_feature)         # N x 2 Wahrscheinlichkeiten
    m = Categorical(prob)                   # Categorical-Verteilung
    a = m.sample()                          # 0 oder 1 pro Subgraph
    take_action = (np.sum(a.numpy()) != 0)  # Mindestens einer selektiert?
```

- Wenn KEIN Subgraph selektiert wurde: `take_action = False` -> wird in `grammar_generation.py:92-93` wiederholt
- Log-Probs werden gespeichert in `agent.saved_log_probs[sample_number][iter_num]` fuer REINFORCE

#### Feature Extractor (GNN)
**Datei:** `GCN/feature_extract.py:14-37`

```python
class feature_extractor():
    def extract(self, graph_mol):
        model = GNN_feature(num_layer=5, emb_dim=300, ...)  # 5-layer GIN
        model.from_pretrained(self.pretrained_model_path)     # Laedt vortrainiertes Modell
        node_features = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        return node_features  # N_atoms x 300
```

- Verwendet vortrainiertes GIN von `GCN/model_gin/supervised_contextpred.pth`
- **Paper Appendix B:** "we choose a pretrained graph neural network (Hu et al., 2019) as our feature extractor f(.)."
- **ACHTUNG:** Das Modell wird NICHT finegetuned! (Appendix B: "we do not finetune its parameters during training")
- **Performance-Problem im Code:** Das Modell wird bei JEDEM Aufruf von `get_subg_feature_for_agent()` neu geladen (molecule_graph.py:133)

#### Subgraph-Feature fuer Agent
**Datei:** `private/molecule_graph.py:137-144`

```python
def get_subg_feature_for_agent(self, subgraph):
    nodes_feat = [self.get_nodes_feature(node_id) for node_id in subgraph.subfrags]
    subfrags_feature = np.mean(nodes_feat, axis=0)  # Mean-Pooling ueber Atom-Features
    return subfrags_feature  # 300-dim
```

Plus die 2 Zusatz-Features (in `grammar_generation.py:86-88`):
- `1 - exp(-num_occurance)`: Gesamthaeufigkeit des Subgraphs (ueber alle Molekuele, saturiert bei ~1)
- `num_in_input / len(input_graphs)`: Anteil der Molekuele die diesen Subgraph enthalten

---

## Arbeitspaket 4: REINFORCE Training Loop <a name="arbeitspaket-4"></a>

### Paper-Kontext

**Paper Sec. 4.2, Eq. 3+4 (S.6):**
> Optimierungsziel: max_theta E_X [sum_i lambda_i * M_i(X)]
> Gradient (REINFORCE):
> nabla_theta E[...] approx (1/N) * sum_n sum_i lambda_i * nabla_theta log(p(X)) * M_i(X)
> - X = Sequenz aller Selektionsentscheidungen
> - M_i = Metriken (Diversity, Synthesizability)
> - lambda_i = Gewichte (lambda_1=1 fuer Diversity, lambda_2=2 fuer Syn)
> - N = Anzahl MC-Samples (MCMC_size = 5)

### Code-Ablauf

#### Hauptschleife
**Datei:** `main.py:97-174`

```
Fuer jede Epoche (max_epoches=50):                          # Z.117
    Fuer jedes MCMC-Sample (MCMC_size=5):                   # Z.123
        1. Daten & Grammatik kopieren (deepcopy)             # Z.125-127
        2. MCMC_sampling() -> Grammatik konstruieren         # Z.128
        3. evaluate() -> Diversity + Syn messen              # Z.130
        4. R = diversity + 2 * syn_rate                      # Z.133

    Policy Loss berechnen:                                    # Z.148-157
        returns normalisieren (- mean)                        # Z.149
        Fuer jedes Sample, jede Iteration:
            loss += -log_prob * gamma^(T-t) * R              # Z.157

    Backprop + Optimizer Step                                 # Z.160-162
    agent.saved_log_probs.clear()                             # Z.163
```

#### Reward-Berechnung
**Datei:** `main.py:133`

```python
R = eval_metric['diversity'] + 2 * eval_metric['syn']
```

- **Paper Sec. 5.4 (S.9):** "we use lambda_1 = 1, lambda_2 = 2 for a balanced trade-off between Diversity and RS."
- Figure 4 zeigt den Trade-off fuer verschiedene lambda-Kombinationen

#### Policy Loss im Detail
**Datei:** `main.py:147-157`

```python
returns = torch.tensor(returns)
returns = (returns - returns.mean())  # Baseline-Subtraktion (Varianzreduktion)

for sample_number in agent.saved_log_probs.keys():
    max_iter_num = max(list(agent.saved_log_probs[sample_number].keys()))
    for iter_num_key in agent.saved_log_probs[sample_number].keys():
        log_probs = agent.saved_log_probs[sample_number][iter_num_key]
        for log_prob in log_probs:
            policy_loss += (-log_prob * gamma ** (max_iter_num - iter_num_key) * returns[sample_number]).sum()
```

- **Paper Eq. 4:** "We then apply gradient ascent" -> im Code: `-log_prob * R` (Gradient Descent auf negativen Reward)
- **Discount:** `gamma^(T-t)` mit gamma=0.99 -- fruehere Entscheidungen werden weniger gewichtet
- **Paper:** "Note that M_i(X) is normalized to zero mean for each sampling batch to reduce variance in training."

#### Hyperparameter (Paper Appendix B, S.13)
**Datei:** `main.py:178-191`

| Parameter | Wert | Code-Zeile | Paper |
|-----------|------|------------|-------|
| max_epoches | 50 | Z.182 | App. B: "20 epochs" (Diskrepanz!) |
| MCMC_size | 5 | Z.184 | App. B: "MC sampling size as 5" |
| learning_rate | 1e-2 | Z.185 | App. B: "learning rate 0.01" |
| hidden_size | 128 | Z.181 | App. B: "size 300 and 128" |
| gamma | 0.99 | Z.186 | Nicht explizit im Paper |
| num_generated_samples | 100 | Z.183 | App. B: "10k samples" (Eval vs. Training) |

---

## Arbeitspaket 5: Grammatik-Produktion (Molekuel-Generierung) <a name="arbeitspaket-5"></a>

### Paper-Kontext

**Paper Sec. 3, "Formal Grammar" (S.4):**
> Erzeugung: Starte mit Initial-Symbol X, wende iterativ Regeln an:
> 1. Finde einen NT-Knoten im aktuellen Graphen
> 2. Finde eine Regel deren LHS zum NT-Knoten passt (Subgraph-Matching)
> 3. Ersetze den NT-Knoten durch die RHS der Regel
> 4. Wiederhole bis keine NT-Knoten mehr vorhanden

**Paper Appendix A (S.13), "Molecule Generation":**
> Wahrscheinlichkeit einer Regel r in Iteration t: p(r) = Z^{-1} * exp(alpha * t * x_r)
> - x_r = 1 wenn die Regel eine End-Regel ist (RHS hat nur Terminale)
> - alpha = 0.5
> - Effekt: Mit steigender Iteration werden End-Regeln bevorzugt (damit Generierung terminiert)

### Code
**Datei:** `grammar_generation.py:136-189`

```python
def random_produce(grammar):
```

1. **Start** (Z.157-165):
   - Erstelle leeren Hypergraphen
   - Waehle zufaellig eine Start-Regel und wende sie an

2. **Iterative Expansion** (Z.166-182):
   - Fuer JEDE Regel in der Grammatik: teste ob sie anwendbar ist (`graph_rule_applied_to`)
   - Berechne Wahrscheinlichkeiten mit `prob_schedule()` (Z.143-155)
   - Sample eine Regel gewichtet nach Wahrscheinlichkeit
   - Abbruch wenn: nur noch Start-Regeln passen ODER iter > 30

3. **prob_schedule()** (Z.143-155):
   ```python
   prob = exp(0.5 * iter * x_r)  # x_r = is_ending (0 oder 1)
   ```
   - **Paper Appendix A:** "p(r) = Z^{-1} exp(alpha*t*x_r), alpha=0.5"
   - Start-Regeln bekommen Wahrscheinlichkeit 0 (Z.150)

4. **Hypergraph -> Molekuel** (Z.183-188):
   - `hg_to_mol(hypergraph)` konvertiert zurueck
   - **Datei:** `private/hypergraph.py` -- Funktion `hg_to_mol()`

#### graph_rule_applied_to() -- Regel anwenden
**Datei:** `private/grammar.py:150-300+`

- **Start-Regel** (Z.168-188): Fuegt alle Knoten und Edges der RHS zum Hypergraphen hinzu
- **Expansion-Regel** (Z.190+):
  1. Finde NT-Edges im Hypergraphen die zur LHS passen
  2. Entferne die NT-Edge
  3. Fuege die RHS ein, verbinde Anchor-Knoten korrekt
  - **Paper Sec. 4 (S.4):** "we use subgraph matching (Gentner, 1983) to test whether the current graph contains a subgraph that is isomorphic to the rule's left-hand side."

---

## Arbeitspaket 6: Evaluation & Metriken <a name="arbeitspaket-6"></a>

### Paper-Kontext

**Paper Sec. 5.1 (S.7):**
> Metriken:
> - **Validity:** % chemisch valider Molekuele
> - **Uniqueness:** % einzigartiger Molekuele
> - **Diversity:** Mittlere paarweise Tanimoto-Distanz (Morgan Fingerprints)
> - **Chamfer Distance:** Distanz zwischen generierten und Trainings-Molekuelen
> - **Retro* Score (RS):** Erfolgsrate der retrosynthetischen Planung
> - **Membership:** % Molekuele die zur Monomer-Klasse gehoeren

### Code

#### evaluate()
**Datei:** `main.py:21-62`

1. **Sampling** (Z.31-47): Generiert `num_generated_samples` (100) einzigartige Molekuele via `random_produce()`
   - Stoppt wenn 100 Samples erreicht oder 10 Fehlversuche hintereinander
   - Prueft auf Duplikate via kanonische SMILES

2. **Diversity** (Z.51-53): `div.get_diversity(generated_samples)`
   - **Datei:** `private/metrics.py:8-26`
   - Morgan Fingerprints (Radius=3, 2048 Bits) (Z.19)
   - Paarweise Tanimoto-Similarity, dann `1 - mean_similarity`

3. **Synthesizability** (Z.58-59): `retro_sender(generated_samples, args)`
   - Kommuniziert via File-IPC mit dem Retro*-Prozess

---

## Arbeitspaket 7: Retro-Star Synthesizability <a name="arbeitspaket-7"></a>

### Paper-Kontext

**Paper Sec. 5.1 (S.7):**
> "Retro* Score (RS): Success rate of the Retro* model (Chen et al., 2020)
> which was trained to find a retrosynthesis path to build a molecule
> from a list of commercially available ones."

### Code -- Zwei-Prozess-Architektur

Das System nutzt File-basierte IPC weil Retro* eine separate Conda-Umgebung braucht.

#### Sender (Hauptprozess)
**Datei:** `main.py:65-94`

```python
def retro_sender(generated_samples, args):
    # Schreibt SMILES in sender_file (generated_samples.txt)
    # Wartet auf Ergebnisse in receiver_file (output_syn.txt)
    # Gibt mean(syn_status) zurueck
```

- File-Locking via `fcntl.flock()` (Z.71-77)
- Polling mit `time.sleep(1)` (Z.92)

#### Listener (Separater Prozess)
**Datei:** `retro_star_listener.py:40-87`

```python
def main(proc_id, filename, output_filename):
    syn = Synthesisability()
    while(True):
        # Liest unbearbeitete Molekuele aus sender_file
        # Markiert sie als "working"
        # Fuehrt Retro*-Planung durch
        # Schreibt Ergebnis (True/False) in output_file
```

- Gestartet in separatem Terminal: `conda activate retro_star_env && bash retro_star_listener.sh 1`
- Nutzt `RSPlanner` mit 50 Iterationen, top-k=50 (Z.21-26)

---

## Zusammenfassung: Gesamtablauf von Anfang bis Ende

```
1. SMILES einlesen                    main.py:197-202
2. data_processing()                  grammar_generation.py:13-54
   |-> mol_to_hg()                    private/hypergraph.py
   |-> SubGraphSet erstellen          private/subgraph_set.py
3. learn() Hauptschleife              main.py:97-174
   |
   |-> Epoche (50x)                   main.py:117
   |   |-> MCMC Sample (5x)          main.py:123
   |   |   |-> MCMC_sampling()        grammar_generation.py:120-133
   |   |   |   |-> grammar_generation()  grammar_generation.py:57-117
   |   |   |   |   |-> Agent sampling    agent.py:22-35
   |   |   |   |   |-> merge_selected    molecule_graph.py:193-260
   |   |   |   |   |-> generate_rule()   grammar.py:835-992
   |   |   |   |   |-> update_subgraph() molecule_graph.py:262-296
   |   |   |
   |   |   |-> evaluate()             main.py:21-62
   |   |   |   |-> random_produce()   grammar_generation.py:136-189
   |   |   |   |-> InternalDiversity  private/metrics.py
   |   |   |   |-> retro_sender()     main.py:65-94
   |   |   |
   |   |   |-> R = div + 2*syn        main.py:133
   |   |
   |   |-> Policy Loss berechnen      main.py:147-157
   |   |-> optimizer.step()           main.py:160-162
```

---

## Glossar

| Begriff | Paper | Code |
|---------|-------|------|
| Hyperedge | E_H, verbindet >2 Knoten | `edge` in Hypergraph |
| Anchor Node | V_anc, verbindet Sub- mit Reststruktur | `ext_node` |
| Non-Terminal (NT) | R*, Platzhalter fuer noch zu expandierende Teile | `NTSymbol` |
| Terminal | Atome, Bonds | `TSymbol`, `BondSymbol` |
| Production Rule | p_i : LHS -> RHS | `ProductionRule(lhs, rhs)` |
| Potential Function | F_theta | `Agent` (MLP) |
| Feature Extractor | f(e) | `feature_extractor` (GIN) |
| MSF | Minimum Spanning Forest | Bottom-up Kontraktionsprozess |
| MCMC | Monte Carlo Sampling | `MCMC_sampling()`, `MCMC_size=5` |

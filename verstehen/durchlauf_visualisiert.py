#!/usr/bin/env python
"""
=============================================================================
DIDAKTISCHES DURCHLAUF-SKRIPT: DEG Paper Step-by-Step
=============================================================================

Dieses Skript zeigt den GESAMTEN Ablauf des DEG-Papers anhand von 2 kleinen
Isocyanat-Molekuelen. Jeder Schritt wird ausfuehrlich gedruckt/visualisiert.

Ablauf:
  Phase 1: Datenverarbeitung (Molekuel -> Hypergraph -> Subgraphen)
  Phase 2: Eine MCMC-Iteration (Bottom-Up Grammatik-Konstruktion)
  Phase 3: Kompletter MCMC-Durchlauf (alle Iterationen bis fertig)
  Phase 4: Warum 5 MCMC Samples? (Vergleich verschiedener Samples)
  Phase 5: REINFORCE Training (eine Epoche)
  Phase 6: Molekuel-Generierung aus der gelernten Grammatik

=============================================================================
"""

import sys
import os

# Repo-Root zum Path hinzufuegen
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, DataStructs
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim

from private import *
from grammar_generation import data_processing, grammar_generation, MCMC_sampling, random_produce
from agent import Agent, sample

# Bilder speichern in verstehen/bilder/
IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bilder")
os.makedirs(IMG_DIR, exist_ok=True)


# ============================================================================
# HILFSFUNKTIONEN FUER VISUALISIERUNG
# ============================================================================

def banner(text, char="="):
    """Druckt einen sichtbaren Banner."""
    line = char * 70
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}\n")


def print_mol_info(smiles, label=""):
    """Druckt Basis-Infos ueber ein Molekuel."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  [{label}] UNGUELTIG: {smiles}")
        return
    print(f"  [{label}]")
    print(f"    SMILES:     {smiles}")
    print(f"    #Atome:     {mol.GetNumAtoms()}")
    print(f"    #Bindungen: {mol.GetNumBonds()}")
    # Ringe
    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()
    print(f"    #Ringe:     {len(rings)}")
    for i, ring in enumerate(rings):
        atoms_in_ring = [mol.GetAtomWithIdx(a).GetSymbol() for a in ring]
        print(f"      Ring {i}: Atome {list(ring)} = {atoms_in_ring}")


def save_mol_image(mol, filename, label=""):
    """Speichert ein Molekuel-Bild als PNG."""
    if mol is None:
        return
    path = os.path.join(IMG_DIR, filename)
    img = Draw.MolToImage(mol, size=(400, 300))
    img.save(path)
    print(f"    -> Bild gespeichert: bilder/{filename}")


def print_hypergraph_info(hg, label=""):
    """Druckt die Struktur eines Hypergraphen."""
    print(f"  Hypergraph [{label}]:")
    print(f"    #Knoten (=Bonds): {hg.num_nodes}  |  #Edges (=Atome/Hyperedges): {hg.num_edges}")

    print(f"    Edges (= Atome/Gruppen):")
    for edge in sorted(hg.edges):
        attr = hg.edge_attr(edge)
        symbol = attr.get('symbol', '?')
        terminal = attr.get('terminal', '?')
        visited = attr.get('visited', False)
        nodes = hg.nodes_in_edge(edge)
        node_strs = []
        for n in sorted(nodes):
            n_attr = hg.node_attr(n)
            bond_sym = n_attr.get('symbol', '?')
            node_strs.append(f"{n}(bond_type={bond_sym.bond_type})")
        t_marker = "T" if terminal else "NT"
        v_marker = " [visited]" if visited else ""
        print(f"      {edge}: {symbol.symbol if hasattr(symbol, 'symbol') else symbol} "
              f"({t_marker}){v_marker} -- Knoten: {', '.join(node_strs)}")


def print_subgraphs(input_graph, label=""):
    """Druckt die aktuellen Subgraphen eines InputGraphs."""
    print(f"  Subgraphen von [{label}]: {len(input_graph.subgraphs)} Stueck")
    for i, (subg, idx) in enumerate(zip(input_graph.subgraphs, input_graph.subgraphs_idx)):
        sml = Chem.MolToSmiles(subg.mol)
        print(f"    Subgraph {i}: Atom-Indices={idx}  SMILES={sml}  #Atome={subg.mol.GetNumAtoms()}")


def print_subgraph_set(subgraph_set):
    """Druckt die SubGraphSet-Statistiken (welche Subgraphen kommen wo vor)."""
    print(f"  SubGraphSet: {len(subgraph_set.map_to_input)} unique Subgraphen")
    for key, (input_dict, count) in subgraph_set.map_to_input.items():
        n_inputs = len(input_dict.keys())
        print(f"    '{key.sml}': kommt {count}x vor, in {n_inputs} Molekuel(en)")


def print_grammar(grammar, label=""):
    """Druckt alle Regeln der Grammatik."""
    print(f"  Grammatik [{label}]: {grammar.num_prod_rule} Regeln")
    for i, rule in enumerate(grammar.prod_rule_list):
        rule_type = "START" if rule.is_start_rule else ("END" if rule.is_ending else "EXPANSION")
        lhs_str = "X (Initial)" if rule.is_start_rule else f"NT(degree={rule.lhs.num_nodes})"

        # RHS Zusammenfassung
        rhs_terminal_edges = [e for e in rule.rhs.edges if rule.rhs.edge_attr(e).get('terminal', False)]
        rhs_nt_edges = [e for e in rule.rhs.edges if not rule.rhs.edge_attr(e).get('terminal', True)]
        rhs_atoms = [rule.rhs.edge_attr(e)['symbol'].symbol for e in rhs_terminal_edges]

        print(f"    Regel {i} [{rule_type}]: {lhs_str} -> "
              f"RHS({len(rhs_terminal_edges)} Atome: {rhs_atoms}, {len(rhs_nt_edges)} NT-Edges)")


def print_action_explanation(action_list, subgraphs, subgraphs_idx):
    """Erklaert was der Agent entschieden hat."""
    print(f"  Agent-Entscheidung (0=ignorieren, 1=selektieren):")
    for i, (action, subg, idx) in enumerate(zip(action_list, subgraphs, subgraphs_idx)):
        sml = Chem.MolToSmiles(subg.mol)
        marker = ">>> SELEKTIERT <<<" if action == 1 else "    ignoriert"
        print(f"    Subgraph {i}: action={action}  {marker}  Indices={idx}  SMILES={sml}")


# ============================================================================
# PHASE 1: DATENVERARBEITUNG
# ============================================================================

def phase1_datenverarbeitung():
    banner("PHASE 1: DATENVERARBEITUNG", "=")
    print("""
    Was passiert hier? (Paper Sec. 3, S.3)
    ----------------------------------------
    1. Wir nehmen 2 Isocyanat-Molekuele als Eingabe
    2. Jedes Molekuel wird als Hypergraph dargestellt
    3. Die Hyperedges (= Cluster) werden identifiziert:
       - Einzelne Bindungen -> je eine Hyperedge
       - Ringe -> je eine Hyperedge die alle Ring-Atome verbindet
    4. Ein SubGraphSet trackt identische Subgraphen ueber alle Molekuele
    """)

    # Zwei einfache Isocyanate waehlen
    smiles_list = [
        "O=C=NCCCCCCN=C=O",    # HDI (Hexamethylene diisocyanate) - kein Ring, einfach
        "CC1=C(C=C(C=C1)CN=C=O)N=C=O",  # TDI (Toluene diisocyanate) - mit Ring
    ]

    print("--- Schritt 1.1: Eingabe-Molekuele ---")
    for i, sml in enumerate(smiles_list):
        print_mol_info(sml, f"Molekuel {i}")
        mol = Chem.MolFromSmiles(sml)
        save_mol_image(mol, f"phase1_mol{i}.png", f"Molekuel {i}")
    print()

    print("--- Schritt 1.2: data_processing() aufrufen ---")
    print("    Konvertiert SMILES -> Kekulized Mol -> Cluster finden -> Hypergraph")
    print()

    GNN_model_path = os.path.join(REPO_ROOT, "GCN/model_gin/supervised_contextpred.pth")
    subgraph_set, input_graphs_dict = data_processing(smiles_list, GNN_model_path, motif=False)

    print()
    print("--- Schritt 1.3: Ergebnis der Datenverarbeitung ---")
    print()

    for i, (key, graph) in enumerate(input_graphs_dict.items()):
        print(f"  === Molekuel {i}: {Chem.MolToSmiles(graph.mol)} ===")
        print_hypergraph_info(graph.hypergraph, f"Mol {i}")
        print()
        print_subgraphs(graph, f"Mol {i}")
        print()

    print("--- Schritt 1.4: SubGraphSet (molekueluebergreifend) ---")
    print("""
    Das SubGraphSet zaehlt fuer JEDEN unique Subgraphen:
    - In wie vielen Molekuelen kommt er vor?
    - Wie oft insgesamt?
    Diese Zahlen werden spaeter als Features fuer den Agent benutzt!
    (Paper Sec. 4.2: die 2 Zusatz-Features neben den 300-dim GNN-Features)
    """)
    print_subgraph_set(subgraph_set)
    print()

    return smiles_list, subgraph_set, input_graphs_dict, GNN_model_path


# ============================================================================
# PHASE 2: EINE MCMC-ITERATION (Bottom-Up Konstruktion)
# ============================================================================

def phase2_eine_iteration(input_graphs_dict, subgraph_set, GNN_model_path):
    banner("PHASE 2: EINE MCMC-ITERATION (Grammatik-Konstruktion)", "=")
    print("""
    Was passiert in EINER Iteration? (Paper Sec. 4.1, S.5, Figure 3)
    ------------------------------------------------------------------
    Fuer JEDES Molekuel gleichzeitig:
    1. Der Agent bekommt Features fuer jeden Subgraphen
    2. Der Agent entscheidet: welche Subgraphen sollen kontrahiert werden?
    3. Ueberlappende selektierte Subgraphen werden zusammengefuegt
    4. Fuer jeden zusammengefuegten Subgraph wird eine Produktionsregel erzeugt
    5. Der Hypergraph wird aktualisiert (kontrahierte Teile -> NT-Knoten)

    Danach hat jedes Molekuel WENIGER Subgraphen, und die Grammatik hat MEHR Regeln.
    """)

    # Agent mit zufaelligen Gewichten
    agent = Agent(feat_dim=300, hidden_size=128)
    grammar = ProductionRuleCorpus()

    # Kopien machen (wie im echten Code)
    l_input_graphs_dict = deepcopy(input_graphs_dict)
    l_subgraph_set = deepcopy(subgraph_set)
    l_grammar = deepcopy(grammar)

    print("--- VOR der Iteration ---")
    for i, (key, graph) in enumerate(l_input_graphs_dict.items()):
        print_subgraphs(graph, f"Mol {i}")
    print()
    print_grammar(l_grammar, "vor Iteration")
    print()

    # EINE Iteration ausfuehren
    print("--- Iteration 0 ausfuehren ---")
    print()

    # Wir machen grammar_generation manuell, um mehr zu printen
    mcmc_iter = 0
    sample_number = 0

    class FakeArgs:
        pass
    args = FakeArgs()

    plist = [*l_subgraph_set.map_to_input]
    print(f"  Noch {len(plist)} unique Subgraphen im SubGraphSet -> weiter gehts!")
    print()

    org_input_graphs_dict = deepcopy(l_input_graphs_dict)
    org_subgraph_set = deepcopy(l_subgraph_set)
    org_grammar = deepcopy(l_grammar)

    l_input_graphs_dict = deepcopy(org_input_graphs_dict)
    l_subgraph_set = deepcopy(org_subgraph_set)
    l_grammar = deepcopy(org_grammar)

    for i, (key, input_g) in enumerate(l_input_graphs_dict.items()):
        print(f"  === Verarbeite Molekuel {i} ===")
        print_subgraphs(input_g, f"Mol {i}")
        print()

        action_list = []
        all_final_features = []

        if len(input_g.subgraphs) > 1:
            print(f"    Schritt 2a: Feature-Berechnung fuer {len(input_g.subgraphs)} Subgraphen")
            for subgraph, subgraph_idx in zip(input_g.subgraphs, input_g.subgraphs_idx):
                subg_feature = input_g.get_subg_feature_for_agent(subgraph)
                num_occurance = l_subgraph_set.map_to_input[MolKey(subgraph)][1]
                num_in_input = len(l_subgraph_set.map_to_input[MolKey(subgraph)][0].keys())
                final_feature = []
                final_feature.extend(subg_feature.tolist())
                final_feature.append(1 - np.exp(-num_occurance))
                final_feature.append(num_in_input / len(list(l_input_graphs_dict.keys())))
                all_final_features.append(torch.unsqueeze(torch.from_numpy(np.array(final_feature)).float(), 0))

                sml = Chem.MolToSmiles(subgraph.mol)
                print(f"      Subgraph '{sml}': GNN-Feature(300-dim) + "
                      f"freq={1 - np.exp(-num_occurance):.3f} + "
                      f"coverage={num_in_input}/{len(list(l_input_graphs_dict.keys()))}")

            print()
            print(f"    Schritt 2b: Agent-Sampling (MLP entscheidet fuer jeden Subgraph: 0 oder 1)")

            # Sample bis mindestens ein Subgraph selektiert wird
            attempt = 0
            while True:
                action_list, take_action = sample(agent, torch.vstack(all_final_features), mcmc_iter, sample_number)
                attempt += 1
                if take_action:
                    break
                if attempt > 50:
                    # Fallback: waehle den ersten
                    action_list = np.array([1] + [0] * (len(input_g.subgraphs) - 1))
                    break

            print()
            print_action_explanation(action_list, input_g.subgraphs, input_g.subgraphs_idx)
            print()

        elif len(input_g.subgraphs) == 1:
            action_list = [1]
            print(f"    Nur 1 Subgraph uebrig -> automatisch selektiert")
        else:
            print(f"    Keine Subgraphen -> uebersprungen")
            continue

        # Merge
        print(f"    Schritt 2c: Zusammenfuegen ueberlappender selektierter Subgraphen")
        p_star_list = input_g.merge_selected_subgraphs(action_list)
        for j, p_star in enumerate(p_star_list):
            sml = Chem.MolToSmiles(p_star.mol)
            print(f"      Merged Subgraph {j}: SMILES={sml}  Indices={p_star.subfrags}")

        print()
        print(f"    Schritt 2d: Produktionsregeln generieren")

        for p_star in p_star_list:
            is_inside, subgraphs, subgraphs_idx = input_g.is_candidate_subgraph(p_star)
            if is_inside:
                for subg, subg_idx in zip(subgraphs, subgraphs_idx):
                    if subg_idx not in input_g.subgraphs_idx:
                        continue
                    n_rules_before = l_grammar.num_prod_rule
                    l_grammar = generate_rule(input_g, subg, l_grammar)
                    n_rules_after = l_grammar.num_prod_rule
                    if n_rules_after > n_rules_before:
                        rule = l_grammar.prod_rule_list[-1]
                        rule_type = "START" if rule.is_start_rule else ("END" if rule.is_ending else "EXPANSION")
                        print(f"      -> NEUE Regel {n_rules_after-1} [{rule_type}] erzeugt!")
                    else:
                        print(f"      -> Regel existiert bereits (Duplikat)")
                    input_g.update_subgraph(subg_idx)

        print()
        print(f"    Nach Iteration -- verbleibende Subgraphen:")
        print_subgraphs(input_g, f"Mol {i}")
        print()

    # Update SubGraphSet
    l_subgraph_set.update([g for (k, g) in l_input_graphs_dict.items()])

    print("--- NACH der Iteration ---")
    print_grammar(l_grammar, "nach 1 Iteration")
    print()
    print_subgraph_set(l_subgraph_set)
    print()

    return l_grammar, l_input_graphs_dict, l_subgraph_set, agent


# ============================================================================
# PHASE 3: KOMPLETTER MCMC-DURCHLAUF
# ============================================================================

def phase3_kompletter_mcmc(input_graphs_dict, subgraph_set, GNN_model_path):
    banner("PHASE 3: KOMPLETTER MCMC-DURCHLAUF (alle Iterationen)", "=")
    print("""
    Was passiert hier? (Paper Sec. 4.1, S.5)
    ------------------------------------------
    Der MCMC-Durchlauf wiederholt die Iterationen aus Phase 2, bis ALLE
    Hyperedges in ALLEN Molekuelen kontrahiert sind.

    Am Ende: Jedes Molekuel ist zu einem einzigen NT-Knoten geschrumpft,
    und die Grammatik enthaelt alle noetiegen Regeln um die Molekuele
    (und neue!) zu erzeugen.
    """)

    agent = Agent(feat_dim=300, hidden_size=128)
    grammar = ProductionRuleCorpus()

    l_input_graphs_dict = deepcopy(input_graphs_dict)
    l_subgraph_set = deepcopy(subgraph_set)
    l_grammar = deepcopy(grammar)

    class FakeArgs:
        pass
    args = FakeArgs()

    iter_num = 0
    while True:
        print(f"  --- MCMC Iteration {iter_num} ---")

        # Wie viele Subgraphen gibt es noch?
        total_subgraphs = sum(len(g.subgraphs) for k, g in l_input_graphs_dict.items())
        print(f"    Verbleibende Subgraphen insgesamt: {total_subgraphs}")

        done_flag, new_igd, new_ss, new_g = grammar_generation(
            agent, l_input_graphs_dict, l_subgraph_set, l_grammar,
            iter_num, 0, args  # sample_number=0
        )

        print(f"    Fertig? {done_flag}")
        print(f"    Grammatik hat jetzt {new_g.num_prod_rule} Regeln")

        if done_flag:
            print()
            print("    ALLE Hyperedges kontrahiert -- Grammatik-Konstruktion abgeschlossen!")
            break

        l_input_graphs_dict = deepcopy(new_igd)
        l_subgraph_set = deepcopy(new_ss)
        l_grammar = deepcopy(new_g)
        iter_num += 1
        print()

    print()
    print_grammar(new_g, "fertige Grammatik")
    print()

    return new_g, agent


# ============================================================================
# PHASE 4: WARUM 5 MCMC SAMPLES?
# ============================================================================

def phase4_mcmc_samples(input_graphs_dict, subgraph_set, GNN_model_path):
    banner("PHASE 4: WARUM 5 MCMC SAMPLES?", "=")
    print("""
    Kernfrage: Warum fuehren wir den Durchlauf 5x durch? (Paper Sec. 4.2, S.6)
    ---------------------------------------------------------------------------
    Der Agent entscheidet per Bernoulli-Sampling welche Subgraphen kontrahiert
    werden. Das ist STOCHASTISCH -- verschiedene Durchlaeufe erzeugen
    VERSCHIEDENE Grammatiken!

    Die 5 Samples pro Epoche dienen dazu:
    1. Verschiedene Grammatiken zu erzeugen (Exploration)
    2. Jede Grammatik wird evaluiert (Diversity + Synthesizability)
    3. Der Reward wird gemittelt/normalisiert (Varianzreduktion)
    4. Der Agent lernt: welche Entscheidungen fuehren zu guten Grammatiken?

    """)

    agent = Agent(feat_dim=300, hidden_size=128)

    class FakeArgs:
        pass
    args = FakeArgs()

    n_samples = 3  # 3 statt 5 fuer Geschwindigkeit
    results = []

    for sample_num in range(n_samples):
        print(f"  === MCMC Sample {sample_num} ===")

        grammar = ProductionRuleCorpus()
        l_igd = deepcopy(input_graphs_dict)
        l_ss = deepcopy(subgraph_set)
        l_g = deepcopy(grammar)

        iter_num, final_grammar, final_igd = MCMC_sampling(
            agent, l_igd, l_ss, l_g, sample_num, args
        )

        print(f"    Iterationen: {iter_num}")
        print(f"    Regeln: {final_grammar.num_prod_rule}")

        # Generiere ein paar Molekuele
        generated = []
        for _ in range(20):
            mol, _ = random_produce(final_grammar)
            if mol is not None:
                sml = Chem.CanonSmiles(Chem.MolToSmiles(mol))
                if sml not in [Chem.CanonSmiles(Chem.MolToSmiles(m)) for m in generated]:
                    generated.append(mol)
            if len(generated) >= 5:
                break

        print(f"    Generierte Molekuele ({len(generated)}):")
        for mol in generated:
            print(f"      {Chem.MolToSmiles(mol)}")

        # Diversity berechnen
        if len(generated) >= 2:
            div = InternalDiversity()
            diversity = div.get_diversity(generated)
            print(f"    Diversity: {diversity:.4f}")
        else:
            diversity = 0.0
            print(f"    Diversity: nicht berechenbar (zu wenige Molekuele)")

        results.append({
            'sample': sample_num,
            'n_rules': final_grammar.num_prod_rule,
            'n_generated': len(generated),
            'diversity': diversity,
            'grammar': final_grammar,
        })
        print()

    print("  --- Vergleich der 3 Samples ---")
    print(f"  {'Sample':<8} {'#Regeln':<10} {'#Generiert':<12} {'Diversity':<10}")
    print(f"  {'-'*40}")
    for r in results:
        print(f"  {r['sample']:<8} {r['n_rules']:<10} {r['n_generated']:<12} {r['diversity']:<10.4f}")
    print()
    print("  -> Jedes Sample erzeugt eine ANDERE Grammatik mit anderen Eigenschaften!")
    print("     Der REINFORCE-Algorithmus nutzt diese Varianz zum Lernen.")
    print()

    return results


# ============================================================================
# PHASE 5: REINFORCE TRAINING (eine Epoche)
# ============================================================================

def phase5_reinforce(input_graphs_dict, subgraph_set, GNN_model_path):
    banner("PHASE 5: REINFORCE TRAINING (eine Epoche)", "=")
    print("""
    Was passiert hier? (Paper Sec. 4.2, Eq. 3+4, S.6)
    ---------------------------------------------------
    Eine Epoche des Trainings:
    1. Fuer N=3 MCMC-Samples: Grammatik konstruieren + evaluieren
    2. Reward R = diversity + 2 * syn_rate berechnen
       (Wir nutzen einen Dummy fuer syn_rate)
    3. Returns normalisieren (- mean) als Baseline
    4. Policy Loss = -sum(log_prob * gamma^(T-t) * R)
    5. Backpropagation + Optimizer Step

    Das ist REINFORCE (Williams, 1992): der Agent lernt welche
    Selektions-Entscheidungen zu hohen Rewards fuehren.
    """)

    agent = Agent(feat_dim=300, hidden_size=128)
    optimizer = optim.Adam(agent.parameters(), lr=1e-2)

    class FakeArgs:
        pass
    args = FakeArgs()

    MCMC_SIZE = 3
    GAMMA = 0.99

    returns = []
    log_returns = []

    for sample_num in range(MCMC_SIZE):
        print(f"  === Sample {sample_num}/{MCMC_SIZE} ===")

        grammar = ProductionRuleCorpus()
        l_igd = deepcopy(input_graphs_dict)
        l_ss = deepcopy(subgraph_set)
        l_g = deepcopy(grammar)

        iter_num, final_grammar, final_igd = MCMC_sampling(
            agent, l_igd, l_ss, l_g, sample_num, args
        )

        # Evaluierung (vereinfacht, ohne Retro*)
        generated = []
        for _ in range(30):
            mol, _ = random_produce(final_grammar)
            if mol is not None:
                sml = Chem.CanonSmiles(Chem.MolToSmiles(mol))
                if sml not in [Chem.CanonSmiles(Chem.MolToSmiles(m)) for m in generated]:
                    generated.append(mol)
            if len(generated) >= 10:
                break

        # Diversity
        if len(generated) >= 2:
            div = InternalDiversity()
            diversity = div.get_diversity(generated)
        else:
            diversity = 0.0

        # Dummy syn_rate (statt Retro*)
        syn_rate = np.random.uniform(0.1, 0.4)

        R = diversity + 2 * syn_rate
        returns.append(R)
        log_returns.append({'diversity': diversity, 'syn': syn_rate})

        print(f"    Regeln: {final_grammar.num_prod_rule}  |  "
              f"Generiert: {len(generated)}  |  "
              f"Diversity: {diversity:.4f}  |  "
              f"Syn(dummy): {syn_rate:.4f}  |  "
              f"R = {R:.4f}")

    print()
    print("  --- Policy Loss Berechnung ---")

    returns_tensor = torch.tensor(returns)
    returns_normalized = returns_tensor - returns_tensor.mean()

    print(f"  Rohe Returns:        {[f'{r:.4f}' for r in returns]}")
    print(f"  Normalisierte Returns: {[f'{r:.4f}' for r in returns_normalized.tolist()]}")
    print(f"    (Subtraktion des Mittelwerts = Baseline fuer Varianzreduktion)")
    print()

    # Policy Loss berechnen
    policy_loss = torch.tensor([0.])
    n_decisions = 0

    for sample_number in agent.saved_log_probs.keys():
        max_iter_num = max(list(agent.saved_log_probs[sample_number].keys()))
        for iter_num_key in agent.saved_log_probs[sample_number].keys():
            log_probs = agent.saved_log_probs[sample_number][iter_num_key]
            for log_prob in log_probs:
                discount = GAMMA ** (max_iter_num - iter_num_key)
                contribution = (-log_prob * discount * returns_normalized[sample_number]).sum()
                policy_loss += contribution
                n_decisions += 1

    print(f"  Gesamte Entscheidungen: {n_decisions}")
    print(f"  Policy Loss: {policy_loss.item():.6f}")
    print()

    # Backprop
    print("  --- Backpropagation ---")
    weights_before = agent.affine1.weight.data[0, :3].clone()
    print(f"  Gewichte vorher (erste 3): {weights_before.tolist()}")

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    agent.saved_log_probs.clear()

    weights_after = agent.affine1.weight.data[0, :3].clone()
    print(f"  Gewichte nachher (erste 3): {weights_after.tolist()}")
    print(f"  Differenz:                  {(weights_after - weights_before).tolist()}")
    print()
    print("  -> Die Gewichte haben sich geaendert! Der Agent lernt.")
    print("     Nach vielen Epochen wird er bessere Selektions-Entscheidungen treffen.")
    print()

    return agent


# ============================================================================
# PHASE 6: MOLEKUEL-GENERIERUNG
# ============================================================================

def phase6_generierung(grammar):
    banner("PHASE 6: MOLEKUEL-GENERIERUNG AUS DER GRAMMATIK", "=")
    print("""
    Was passiert hier? (Paper Sec. 3, S.4 + Appendix A, S.13)
    -----------------------------------------------------------
    Die gelernte Grammatik wird wie eine formale Grammatik benutzt:
    1. Starte mit dem Initial-Symbol X (leerer Hypergraph)
    2. Waehle eine Start-Regel und wende sie an
    3. Iterativ: finde NT-Knoten, waehle passende Regel, ersetze
    4. Stoppe wenn keine NT-Knoten mehr vorhanden

    Wichtig: End-Regeln werden mit steigender Iteration wahrscheinlicher
    (prob = exp(0.5 * iter * is_ending)), damit die Generierung terminiert.
    """)

    print("  Verfuegbare Grammatik:")
    print_grammar(grammar, "fuer Generierung")
    print()

    print("  --- Generierung von 10 Molekuelen ---")
    print()

    generated_mols = []
    generated_smiles = []

    for attempt in range(50):
        mol, n_iters = random_produce(grammar)
        if mol is not None:
            sml = Chem.CanonSmiles(Chem.MolToSmiles(mol))
            if sml not in generated_smiles:
                generated_mols.append(mol)
                generated_smiles.append(sml)
                print(f"  Molekuel {len(generated_mols)}: {sml}  (in {n_iters} Iterationen)")
        if len(generated_mols) >= 10:
            break

    print()
    print(f"  -> {len(generated_mols)} einzigartige Molekuele generiert!")
    print()

    # Diversity
    if len(generated_mols) >= 2:
        div = InternalDiversity()
        diversity = div.get_diversity(generated_mols)
        print(f"  Diversity (Tanimoto): {diversity:.4f}")

    # Bild speichern
    if generated_mols:
        try:
            img = Draw.MolsToGridImage(generated_mols[:8], molsPerRow=4, subImgSize=(300, 200))
            img.save(os.path.join(IMG_DIR, "phase6_generated.png"))
            print(f"  -> Bild gespeichert: bilder/phase6_generated.png")
        except Exception as e:
            print(f"  (Bild-Speicherung fehlgeschlagen: {e})")

    return generated_mols


# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

if __name__ == "__main__":
    banner("DEG PAPER: DIDAKTISCHER DURCHLAUF", "#")
    print("""
    Dieses Skript fuehrt den GESAMTEN DEG-Algorithmus Schritt fuer Schritt
    durch, mit nur 2 Molekuelen und ausfuehrlichen Prints.

    Phasen:
      1. Datenverarbeitung (SMILES -> Hypergraph -> Subgraphen)
      2. Eine einzelne MCMC-Iteration im Detail
      3. Kompletter MCMC-Durchlauf (alle Iterationen)
      4. Warum 5 MCMC Samples? (Vergleich)
      5. REINFORCE Training (eine Epoche)
      6. Molekuel-Generierung aus der Grammatik

    Bilder werden in verstehen/bilder/ gespeichert.
    """)

    # Phase 1
    smiles_list, subgraph_set, input_graphs_dict, GNN_model_path = phase1_datenverarbeitung()

    # Phase 2
    grammar_1iter, igd_1iter, ss_1iter, agent_1iter = phase2_eine_iteration(
        input_graphs_dict, subgraph_set, GNN_model_path
    )

    # Phase 3
    full_grammar, agent_full = phase3_kompletter_mcmc(
        input_graphs_dict, subgraph_set, GNN_model_path
    )

    # Phase 4
    sample_results = phase4_mcmc_samples(
        input_graphs_dict, subgraph_set, GNN_model_path
    )

    # Phase 5
    trained_agent = phase5_reinforce(
        input_graphs_dict, subgraph_set, GNN_model_path
    )

    # Phase 6 (benutze die beste Grammatik aus Phase 4)
    best_grammar = max(sample_results, key=lambda r: r['diversity'])['grammar']
    generated_mols = phase6_generierung(best_grammar)

    banner("FERTIG!", "#")
    print("""
    Zusammenfassung:
    ================
    Phase 1: SMILES -> Hypergraph -> Subgraphen (Datenrepraesentation)
    Phase 2: Eine Iteration = Agent selektiert + Regeln extrahiert
    Phase 3: Wiederhole bis alles kontrahiert = fertige Grammatik
    Phase 4: 5 Samples = 5 verschiedene stochastische Grammatiken
    Phase 5: REINFORCE = lerne welche Entscheidungen gut sind
    Phase 6: Grammatik anwenden = neue Molekuele generieren

    Im echten Training:
    - 50 Epochen (jede mit 5 Samples)
    - 100 generierte Molekuele pro Evaluation
    - Retro* fuer echte Synthesizability
    - Reward = diversity + 2 * syn_rate

    Bilder findest du in: verstehen/bilder/
    """)

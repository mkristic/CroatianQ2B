"""
Nakon definiranog skupa logickih upita - 4. korak: priprema dataseta za Q2B
Ovdje je korak 4c.

4c) Priprema dataseta za treniranje: train/valid/test podjela + negative sampling za svaki tip upita generiran u fazi 4b (train_q2b).

Ulaz:  data/queries/logical_queries.pkl  (izlaz logical_query_generator.py)
Izlaz: data/queries/dataset_split.pkl s train/valid/test listama, gdje svaki primjer ima i pozitivne (answers) i negativne odgovore.
"""

import os
import pickle
import random
from typing import Dict, List, Set

def load_queries(pkl_path: str) -> Dict[str, list]:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

# dijeli listu upita na train/valid/test - jednostavan random split po upitima (ne po pojedinacnim odgovorima)
def split_train_valid_test(queries: list, train_ratio: float = 0.8,
                            valid_ratio: float = 0.1, seed: int = 42):
    rng = random.Random(seed)
    shuffled = queries[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train = shuffled[:n_train]
    valid = shuffled[n_train:n_train + n_valid]
    test = shuffled[n_train + n_valid:]

    return train, valid, test

# uzorkuje entitete koji NISU tocan odgovor na upit - potrebno za margin-based loss (model mora naucit rangirati tocne odgovore iznad netocnih)
def sample_negatives(true_answers: Set[str], all_entities: List[str],
                      n_negatives: int, rng: random.Random) -> List[str]:
    negatives = []
    attempts = 0
    max_attempts = n_negatives * 20  # sigurnosna granica ako je graf malen

    while len(negatives) < n_negatives and attempts < max_attempts:
        attempts += 1
        candidate = rng.choice(all_entities)
        if candidate not in true_answers:
            negatives.append(candidate)

    return negatives

# za svaki tip upita (1p, 2p, 2i...) napravi train/valid/test podjelu i doda negative sampleove svakom primjeru
def build_dataset(queries_by_type: Dict[str, list], all_entities: List[str],
                   n_negatives: int = 5, seed: int = 42) -> Dict[str, Dict[str, list]]:
    rng = random.Random(seed)
    dataset = {}

    for qtype, qlist in queries_by_type.items():
        train, valid, test = split_train_valid_test(qlist, seed=seed)

        dataset[qtype] = {}
        for split_name, split_data in [("train", train), ("valid", valid), ("test", test)]:
            enriched = []
            for item in split_data:
                positives = item["answers"]
                negatives = sample_negatives(positives, all_entities, n_negatives, rng)
                enriched.append({
                    "query": item["query"],
                    "positives": positives,
                    "negatives": negatives,
                })
            dataset[qtype][split_name] = enriched
    return dataset

def save_dataset(dataset: Dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

###############################################################################################################################

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    QUERIES_PKL = os.path.join(base_dir, "..", "..", "data", "queries", "logical_queries.pkl")
    ENTITY2ID_PATH = os.path.join(base_dir, "..", "..", "data", "processed", "entity2id.json")
    OUTPUT_PATH = os.path.join(base_dir, "..", "..", "data", "queries", "dataset_split.pkl")

    import json
    with open(ENTITY2ID_PATH, encoding="utf-8") as f:
        entity2id = json.load(f)
    all_entities = list(entity2id.keys())

    queries_by_type = load_queries(QUERIES_PKL)
    dataset = build_dataset(queries_by_type, all_entities, n_negatives=5)

    print("\n=== VELICINE PODJELE PO TIPU UPITA ===")
    for qtype, splits in dataset.items():
        sizes = {name: len(data) for name, data in splits.items()}
        print(f"{qtype}: {sizes}")

    save_dataset(dataset, OUTPUT_PATH)
    print(f"\nSpremljeno u: {OUTPUT_PATH}")
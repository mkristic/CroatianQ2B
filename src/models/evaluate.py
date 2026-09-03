"""
Zadnja faza treniranja: evaluacija treniranog box-embedding modela na test skupu.

Za svaki upit u test splitu, model se pita da rangira SVE entitete u grafu po udaljenosti od kutije upita, te se provjerava na kojem se mjestu (rank)
nalazi stvarni (tocan) odgovor. Sto je rank manji (blizi 1), to bolje.

Standardne metrike za ovakvu evaluaciju (koriste se i u originalnom Query2Box radu):
    MRR     - Mean Reciprocal Rank: prosjek od 1/rank (vise je bolje, max 1.0)
    Hits@1  - udio slucajeva gdje je tocan odgovor bio bas na 1. mjestu
    Hits@3  - udio slucajeva gdje je tocan odgovor bio u top 3
    Hits@10 - udio slucajeva gdje je tocan odgovor bio u top 10

Koristi se "filtered" postavka: kad se rangira jedan tocan odgovor, ostali poznati tocni odgovori za ISTI upit se privremeno uklone iz rangiranja,
da model ne bude nepravedno kaznjen sto je "pogodio" i neki drugi tocan odgovor prije onog trenutno promatranog.
"""

import os
import json
import pickle
import torch
from box_embedding import BoxEmbedding
from train_q2b import answer_query_box

# vraca listu rankova (jedan po pozitivnom odgovoru) za dani upit
def compute_ranks_for_query(model, query, positives, entity2id, relation2id, id2entity, device):
    center, offset = answer_query_box(model, query, entity2id, relation2id, device)

    all_ids = torch.arange(len(entity2id), device=device)
    with torch.no_grad():
        distances = model.box_distance(center, offset, all_ids).squeeze()

    # sortiraj entitete po udaljenosti (manja udaljenost = bolji kandidat)
    sorted_order = torch.argsort(distances).tolist()
    ranked_entities = [id2entity[i] for i in sorted_order]

    positive_set = set(positives)
    ranks = []

    for pos in positives:
        # filtrirano rangiranje: makni ostale tocne odgovore iz liste,
        # osim onog ciji rank trenutno racunamo
        filtered = [e for e in ranked_entities if e == pos or e not in positive_set]
        rank = filtered.index(pos) + 1  # rank pocinje od 1, ne od 0
        ranks.append(rank)

    return ranks


def evaluate(model, test_data_by_type, entity2id, relation2id, id2entity, device):
    results = {}

    for qtype, examples in test_data_by_type.items():
        all_ranks = []

        for example in examples:
            query = example["query"]
            positives = list(example["positives"])
            if not positives:
                continue
            ranks = compute_ranks_for_query(model, query, positives, entity2id, relation2id, id2entity, device)
            all_ranks.extend(ranks)

        if not all_ranks:
            results[qtype] = None
            continue

        mrr = sum(1.0 / r for r in all_ranks) / len(all_ranks)
        hits_1 = sum(1 for r in all_ranks if r <= 1) / len(all_ranks)
        hits_3 = sum(1 for r in all_ranks if r <= 3) / len(all_ranks)
        hits_10 = sum(1 for r in all_ranks if r <= 10) / len(all_ranks)

        results[qtype] = {
            "n_primjera": len(all_ranks),
            "MRR": round(mrr, 4),
            "Hits@1": round(hits_1, 4),
            "Hits@3": round(hits_3, 4),
            "Hits@10": round(hits_10, 4),
        }

    return results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ENTITY2ID_PATH = os.path.join(base_dir, "..", "..", "data", "processed", "entity2id.json")
    RELATION2ID_PATH = os.path.join(base_dir, "..", "..", "data", "processed", "relation2id.json")
    DATASET_PATH = os.path.join(base_dir, "..", "..", "data", "queries", "dataset_split.pkl")
    MODEL_PATH = os.path.join(base_dir, "..", "..", "results", "models", "box_embedding.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(ENTITY2ID_PATH, encoding="utf-8") as f:
        entity2id = json.load(f)
    with open(RELATION2ID_PATH, encoding="utf-8") as f:
        relation2id = json.load(f)
    id2entity = {i: e for e, i in entity2id.items()}

    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    model = BoxEmbedding(
        num_entities=len(entity2id),
        num_relations=len(relation2id),
        embedding_dim=64,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    test_data_by_type = {qtype: splits["test"] for qtype, splits in dataset.items()}
    valid_data_by_type = {qtype: splits["valid"] for qtype, splits in dataset.items()}

    print("=== REZULTATI NA VALID SKUPU ===")
    valid_results = evaluate(model, valid_data_by_type, entity2id, relation2id, id2entity, device)
    for qtype, metrics in valid_results.items():
        print(f"{qtype}: {metrics}")

    print("\n=== REZULTATI NA TEST SKUPU ===")
    test_results = evaluate(model, test_data_by_type, entity2id, relation2id, id2entity, device)
    for qtype, metrics in test_results.items():
        print(f"{qtype}: {metrics}")
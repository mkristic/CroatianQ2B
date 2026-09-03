"""
Nakon definiranog skupa logickih upita - 4. korak: priprema dataseta za Q2B
Ovdje je korak 4b.

4b) Trening petlja za box-embedding model definiran u 4a fazi (box_embedding.py).

Koristi dataset iz dataset_split.py (train/valid/test + negativi) i uci model da kutija upita bude BLIZU pozitivnim odgovorima, a DALEKO od
negativnih (margin-based loss).
"""

import os
import json
import pickle
import random

import torch
import torch.nn.functional as F

from box_embedding import BoxEmbedding


def load_entity_relation_ids(entity2id_path: str, relation2id_path: str):
    with open(entity2id_path, encoding="utf-8") as f:
        entity2id = json.load(f)
    with open(relation2id_path, encoding="utf-8") as f:
        relation2id = json.load(f)
    return entity2id, relation2id

# racuna (center, offset) kutiju upita ovisno o tipu (1p/2p/2i) - ovo je "izvrsavanje" upita kroz model, analogno project()/intersect() funkcijama 
# u LogicalQueryGenerator, ali sad u vektorskom prostoru
def answer_query_box(model: BoxEmbedding, query, entity2id: dict, relation2id: dict, device):
    qtype, structure = query

    if qtype == "1p":
        anchor, (r,) = structure
        anchor_id = torch.tensor([entity2id[anchor]], device=device)
        rel_id = torch.tensor([relation2id[r]], device=device)
        center, offset = model.get_entity_box(anchor_id)
        center, offset = model.project(center, offset, rel_id)
        return center, offset

    if qtype == "2p":
        anchor, (r1, r2) = structure
        anchor_id = torch.tensor([entity2id[anchor]], device=device)
        r1_id = torch.tensor([relation2id[r1]], device=device)
        r2_id = torch.tensor([relation2id[r2]], device=device)
        center, offset = model.get_entity_box(anchor_id)
        center, offset = model.project(center, offset, r1_id)
        center, offset = model.project(center, offset, r2_id)
        return center, offset

    if qtype == "2i":
        (v1, (r1,)), (v2, (r2,)) = structure
        v1_id = torch.tensor([entity2id[v1]], device=device)
        v2_id = torch.tensor([entity2id[v2]], device=device)
        r1_id = torch.tensor([relation2id[r1]], device=device)
        r2_id = torch.tensor([relation2id[r2]], device=device)

        c1, o1 = model.get_entity_box(v1_id)
        c1, o1 = model.project(c1, o1, r1_id)
        c2, o2 = model.get_entity_box(v2_id)
        c2, o2 = model.project(c2, o2, r2_id)

        center, offset = model.intersect([c1, c2], [o1, o2])
        return center, offset

    raise ValueError(f"Nepodrzan tip upita: {qtype}")


def train(model, dataset, entity2id, relation2id, device,
          epochs: int = 20, lr: float = 0.01, margin: float = 1.0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # spoji sve tipove upita u jednu listu primjera za trening
    train_examples = []
    for qtype, splits in dataset.items():
        train_examples.extend(splits["train"])

    for epoch in range(epochs):
        random.shuffle(train_examples)
        total_loss = 0.0
        n_batches = 0

        for example in train_examples:
            query = example["query"]
            positives = list(example["positives"])
            negatives = example["negatives"]

            # preskoci primjere bez pozitiva/negativa (rubni slucajevi)
            if not positives or not negatives:
                continue

            center, offset = answer_query_box(model, query, entity2id, relation2id, device)

            pos_ids = torch.tensor([entity2id[p] for p in positives], device=device)
            neg_ids = torch.tensor([entity2id[n] for n in negatives], device=device)

            pos_dist = model.box_distance(center, offset, pos_ids)
            neg_dist = model.box_distance(center, offset, neg_ids)

            # margin loss: zelimo da su pozitivi BLIZE (manja udaljenost)
            # od negativa za barem 'margin' - inace kaznjavamo model
            loss = F.relu(margin + pos_dist.mean() - neg_dist.mean())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoha {epoch + 1}/{epochs} - prosjecni gubitak: {avg_loss:.4f}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ENTITY2ID_PATH = os.path.join(base_dir, "..", "..", "data", "processed", "entity2id.json")
    RELATION2ID_PATH = os.path.join(base_dir, "..", "..", "data", "processed", "relation2id.json")
    DATASET_PATH = os.path.join(base_dir, "..", "..", "data", "queries", "dataset_split.pkl")
    MODEL_OUTPUT_PATH = os.path.join(base_dir, "..", "..", "results", "models", "box_embedding.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Koristi se uredaj: {device}")

    entity2id, relation2id = load_entity_relation_ids(ENTITY2ID_PATH, RELATION2ID_PATH)

    with open(DATASET_PATH, "rb") as f:
        dataset = pickle.load(f)

    model = BoxEmbedding(
        num_entities=len(entity2id),
        num_relations=len(relation2id),
        embedding_dim=64,
    ).to(device)

    train(model, dataset, entity2id, relation2id, device, epochs=20, lr=0.01)

    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
    print(f"\nModel spremljen u: {MODEL_OUTPUT_PATH}")

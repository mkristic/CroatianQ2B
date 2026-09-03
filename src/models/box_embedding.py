"""
Nakon definiranog skupa logickih upita - 4. korak: priprema dataseta za Q2B
Ovdje je korak 4a.

4a) Box-embedding model - pojednostavljena implementacija Query2Box ideje, skalirana za mali graf i 1p/2p/2i upite.

Svaki entitet i relacija predstavljeni su "kutijom" u vektorskom prostoru:
    center  - sredina kutije (gdje se entitet/vrijednost "nalazi")
    offset  - polu-duljina stranica kutije (koliko je "siroka"/neizvjesna)

Projekcija (p): centar i offset entiteta se POMAKNU i PROSIRE za centar i offset relacije (kutija se "krece" kroz graf).
Presjek (i): kutije se sijeku - novi centar je tezinski prosjek centara, novi offset je min. offset (presjek uvijek suzava kutiju).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoxEmbedding(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        # centri kutija - jedan vektor po entitetu
        self.entity_center = nn.Embedding(num_entities, embedding_dim)

        # offseti entiteta - koliko je kutija "siroka" oko svog centra
        # (softplus kasnije osigurava da je uvijek >= 0)
        self.entity_offset = nn.Embedding(num_entities, embedding_dim)

        # relacije pomicu i siri kutiju: center_offset (pomak sredine) i offset_offset (koliko relacija dodatno prosiruje kutiju)
        self.relation_center = nn.Embedding(num_relations, embedding_dim)
        self.relation_offset = nn.Embedding(num_relations, embedding_dim)

        # inicijalizacija - mali nasumicni brojevi da trening krene stabilno
        nn.init.uniform_(self.entity_center.weight, -1, 1)
        nn.init.uniform_(self.entity_offset.weight, 0, 1)
        nn.init.uniform_(self.relation_center.weight, -1, 1)
        nn.init.uniform_(self.relation_offset.weight, 0, 1)

    # vraca (center, offset) kutiju za zadane entitete
    def get_entity_box(self, entity_ids: torch.Tensor):
        center = self.entity_center(entity_ids)
        offset = F.softplus(self.entity_offset(entity_ids))  # offset uvijek > 0
        return center, offset

    # operator projekcije (p): pomakni centar, prosiri offset
    def project(self, center, offset, relation_ids: torch.Tensor):
        rel_center = self.relation_center(relation_ids)
        rel_offset = F.softplus(self.relation_offset(relation_ids))

        new_center = center + rel_center
        new_offset = offset + rel_offset  # kutija se samo siri kroz projekcije
        return new_center, new_offset

    # operator presjeka (i): kombinira vise kutija u jednu - koristi se za 2i upite
    # centers/offsets su liste tenzora (jedan po uvjetu)
    def intersect(self, centers: list, offsets: list):
        centers_stack = torch.stack(centers, dim=0)   # (broj_uvjeta, batch, dim)
        offsets_stack = torch.stack(offsets, dim=0)

        # tezine za centar - kutije s manjim offsetom (vise "sigurne")
        # dobivaju vecu tezinu u tezinskom prosjeku centara
        weights = F.softmax(-offsets_stack, dim=0)
        new_center = (weights * centers_stack).sum(dim=0)

        # offset presjeka = minimum svih offseta (presjek je uvijek uzi
        # ili jednak najuzoj kutiji koja ulazi u presjek)
        new_offset, _ = offsets_stack.min(dim=0)

        return new_center, new_offset

    # udaljenost izmedu upitne kutije i kandidata za odgovor - manja udaljenost = bolji kandidat
    # koristi se i za trening (loss) i za evaluaciju (rangiranje kandidata)
    def box_distance(self, query_center, query_offset, target_ids: torch.Tensor):
        target_center, _ = self.get_entity_box(target_ids)

        # koliko je target IZVAN kutije upita (bitnije, veca kazna)
        dist_outside = F.relu(torch.abs(target_center - query_center) - query_offset)
        # koliko je target UNUTAR kutije upita (manje bitno, manja kazna)
        dist_inside = F.relu(query_offset - torch.abs(target_center - query_center))
        # kombinacija - alpha < 1 znaci da smo blaziji prema tome sto je vec unutar kutije
        alpha = 0.2
        distance = dist_outside.sum(dim=-1) + alpha * dist_inside.sum(dim=-1)
        return distance

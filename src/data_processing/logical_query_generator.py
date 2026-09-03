"""
    Generator logickih upita za Croatian Query2Box.

    Tipovi upita: 1p, 2p, 2i, 
    gdje je:
        p - projekcija (logicko "postoji")
        i - intersection/presjek (logicko "i").

    Objasnjenje upita: 
        * entiteti (i subjekt i objekt) su u grafu cvorovi v1...vN t.d. je v1 pocetni, a posljednji vN je zavrsni
        - 1p upiti: v1 -> v2
        - 2p upiti: v1 -> v2 -> v3
        - 2i upiti: v1 AND v2 -> v3
"""

import json
import random
import pickle
import os
from collections import defaultdict
from typing import List, Tuple, Set, Dict

class LogicalQueryGenerator:
    def __init__(self, triples_path: str, seed: int = 42):
        self.triples = self._load_triples(triples_path)
        random.seed(seed)
        self._build_indexes()

    # ucitavanje grafa
    def _load_triples(self, path: str) -> List[Tuple[str, str, str]]:
        triples = []

        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 3:
                    triples.append(tuple(parts))
        return triples

    def _build_indexes(self):
        # za 1p i 2p 
        # objects[(s, r)] = objects je skup objekata o t.d. postoji (s, r, o)  
        self.objects: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        # za 2p 
        # entity_relations[s] - skup relacija r koje idu iz s, potrebno za drugi dio upita u 2p upitu
        self.entity_relations: Dict[str, Set[str]] = defaultdict(set)

        # za 2i 
        # subjects[(r, o)] = skup subjekata s t.d. postoji (s, r, o)
        # za ovaj upit se ide obrnutim smjerom o -> r -> s
        #  -> za "koja je osoba rodena u Zagrebu" -> pitamo se "koji entitet/subjekt ima relaciju 'roden' i objekt 'Zagreb'" -> presjek je skup subjekata (osoba)
        self.subjects: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

        for s, r, o in self.triples:
            self.objects[(s, r)].add(o)
            self.entity_relations[s].add(r)
            self.subjects[(r, o)].add(s)

        self.subject_keys = list(self.subjects.keys())

    # projekcija (operator postoji), za 1p i 2p
    def project(self, entities: Set[str], relation: str) -> Set[str]:
        result = set()

        for s in entities:
            result |= self.objects.get((s, relation), set())
        return result

    ###############################################################################################################################
    # UPITI

    # 1p
    def generate_1p(self, n_queries: int, min_answers: int = 1, max_answers: int = 50):
        queries = [] # tu ce ici generirani upiti
        object_candidates = list(self.objects.keys()) # lista tuplova(s, r) -> to je key za objects
        random.shuffle(object_candidates)

        for anchor, relation in object_candidates:
            if len(queries) >= n_queries:
                break
            answers = self.objects[(anchor, relation)]
            if not (min_answers <= len(answers) <= max_answers):
                continue
            query = ("1p", (anchor, (relation,))) # zarez jer je relation isto tuple
            queries.append({"query": query, "answers": set(answers)})

        return queries

    # 2p
    def generate_2p(self, n_queries: int, min_answers: int = 1, max_answers: int = 50, max_attempts: int = 20000):
        queries = []
        seen = set()
        object_candidates = list(self.objects.keys())
        random.shuffle(object_candidates)

        # u p2 su 2 relacije: r1, r2
        attempts = 0
        for anchor, r1 in object_candidates: 
            if len(queries) >= n_queries or attempts >= max_attempts:
                break
            attempts += 1
 
            intermediates = self.objects[(anchor, r1)]
            if not intermediates:
                continue
 
            # koje se relacije mogu nastaviti iz posrednih entiteta
            r2_candidates = set()
            for inter in intermediates:
                r2_candidates |= self.entity_relations.get(inter, set())
            if not r2_candidates:
                continue
 
            r2 = random.choice(list(r2_candidates))
            answers = self.project(intermediates, r2)
            answers.discard(anchor)  # ukloni trivijalan slucaj povratka na sidro
 
            if not (min_answers <= len(answers) <= max_answers):
                continue
 
            key = (anchor, r1, r2)
            if key in seen:
                continue
            seen.add(key)
 
            query = ("2p", (anchor, (r1, r2)))
            queries.append({"query": query, "answers": answers})
 
        return queries

    # 2i
    def generate_2i(self, n_queries: int, min_answers: int = 1, max_answers: int = 50, max_attempts: int = 20000):
        queries = []
        seen = set()
        attempts = 0
 
        # preskoci ako nema dovoljno (r, v) kljuceva za uzorkovanje
        if len(self.subject_keys) < 2:
            return queries
 
        while len(queries) < n_queries and attempts < max_attempts:
            attempts += 1
            (r1, v1), (r2, v2) = random.sample(self.subject_keys, 2)
            if (r1, v1) == (r2, v2):
                continue
 
            s1 = self.subjects[(r1, v1)]
            s2 = self.subjects[(r2, v2)]
            answers = s1 & s2
 
            if not (min_answers <= len(answers) <= max_answers):
                continue
 
            key = frozenset({(r1, v1), (r2, v2)})
            if key in seen:
                continue
            seen.add(key)
 
            query = ("2i", ((v1, (r1,)), (v2, (r2,))))
            queries.append({"query": query, "answers": answers})
 
        return queries



    ###############################################################################################################################
    # citljiv prikaz upit (covjeku citljiv)

    def to_natural_language(self, query) -> str:
        qtype, structure = query

        if qtype == "1p":
            anchor, (r,) = structure
            return f"Koja je vrijednost svojstva '{r}' za entitet '{anchor}'?"

        if qtype == "2p":
            anchor, (r1, r2) = structure
            return (f"Krecuci od '{anchor}', preko relacije '{r1}' pa preko relacije '{r2}' - koji je entitet na kraju?")
 
        if qtype == "2i":
            (v1, (r1,)), (v2, (r2,)) = structure
            return (f"Koji entitet zadovoljava uvjet '{r1}' = '{v1}' i uvjet '{r2}' = '{v2}'?")


        return "Nepoznat tip upita"

###############################################################################################################################
# spremanje rezultata

def save_queries(queries_by_type: Dict[str, list], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # spremanje za kasniji rad s podatcima, machine-readable 
    pkl_path = os.path.join(output_dir, "logical_queries.pkl")

    with open(pkl_path, "wb") as f:
        pickle.dump(queries_by_type, f)

    # spremanje u JSON, human-readable, manji broj upita se sprema cisto za uvid u podatke
    sample = {}

    for qtype, qlist in queries_by_type.items():
        sample[qtype] = [
            {"query": q["query"], "answers": sorted(q["answers"])}
            for q in qlist[:20] # sprema prvih 20
        ]

    json_path = os.path.join(output_dir, "logical_queries_sample.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

    return pkl_path, json_path

###############################################################################################################################
# main

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    TRIPLES_PATH = os.path.join(base_dir, "..", "..", "data", "processed", "croatian_triples.txt")
    OUTPUT_DIR = os.path.join(base_dir, "..", "..", "data", "queries")

    generator = LogicalQueryGenerator(TRIPLES_PATH)

    n_per_type = 300

    queries_by_type = {
        "1p": generator.generate_1p(n_per_type),
        "2p": generator.generate_2p(n_per_type),
        "2i": generator.generate_2i(n_per_type)
    }

    print("\n=== BROJ GENERIRANIH UPITA PO TIPU ===")
    for qtype, qlist in queries_by_type.items():
        print(f"{qtype}: {len(qlist)}")

    print("\n=== PRIMJERI ===")
    for qtype, qlist in queries_by_type.items():
        if qlist:
            example = qlist[0]
            print(f"\n[{qtype}] {generator.to_natural_language(example['query'])}")
            print(f"   Odgovori ({len(example['answers'])}): "
                f"{sorted(example['answers'])[:5]}")

    pkl_path, json_path = save_queries(queries_by_type, OUTPUT_DIR)
    print(f"\nSpremljeno u:\n  {pkl_path}\n  {json_path}")


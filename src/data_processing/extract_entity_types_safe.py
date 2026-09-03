"""
    Pri dodavanju entity_types dictionaryja u croatian_kg_processor.py i ponovnom pokretanju te skripte, ako dode do problema pri izvlacenju podataka s baze (Wikidata vraca 429, 502, 504), 
    postoji opasnost da se croatian_triples.txt overwritea u prazan ili manji file, cime se gube podatci.

    Ovdje se hardkodirano zadaju tipovi entiteta za neke objekte (subjekti vec dobiju tip s Wikidata, a neki objekti onda imaju taj isti tip) i na siguran se nacin spremaju tipovi entiteta
    (bez opasnosti od gubljenja podataka) u 2 koraka:
        1. cita postojece tipove iz croatian_triples.txt i radi backup entity_types.json
        2. safety check: ako je nova verzija prazna ili puno manja od stare -> abortat ce
        3. na postojece tipove dodaje nove
"""

import json
import os
import shutil
from datetime import datetime

# pretvorba predikata is (s, p, o) u tip
# samo ce neki objekti dobiti tip zbog jednostavnosti i intuitivnosti (npr objekti koji predstavljaju koordinate, nadmorsku visinu (brojeve opcenito) i datume - preskoceni)

PREDICATE_TO_OBJECT_TYPE = {
    "drzava": "drzava",
    "drzavljanstvo": "drzava",
    "nalazi_se_u": "lokacija",
    "sjediste": "lokacija",
    "izvor": "lokacija",
    "usce": "lokacija",
    "gorje": "gorje",
    "sluzbeni_jezik": "jezik",
    "mjesto_rodjenja": "lokacija",
    "mjesto_smrti": "lokacija",
    "zanimanje": "zanimanje",
    "bracni_partner": "osoba",
    "obrazovanje": "ustanova",
    "arhitekt": "osoba",
    "arhitektonski_stil": "stil",
    "ravnatelj": "osoba",
    "podrucje_rada": "podrucje",
    "sponzor": "organizacija",
    "organizator": "organizator",
    "sudionik": "sudionik",
    "pisac": "osoba",
    "zanr": "zanr",
    "tema": "tema",
    "nagrada": "nagrada",
    "izdavac": "izdavac",
    "izvodac": "osoba",
    "producent": "osoba",
    "skladatelj": "osoba",
    "svira_instrument": "osoba"
}

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# potrebno dohvatiti trojke da bi se doslo do predikata
def load_triples(path):
    triples = []
    if not os.path.exists(path):
        return triples
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples

# ne mijenja existing_types, vraca novi dictionary
def update_entity_types(existing_types: dict, triples: list) -> dict:
    types = dict(existing_types)

    added = 0
    for s, p, o in triples:
        if not o or o.isdigit():
            continue
        if o in types:
            continue

        # ako tog objekta nema u tipovima
        obj_type = PREDICATE_TO_OBJECT_TYPE.get(p) # p je key tog dictionaryja -> vraca value ili None
        if obj_type:
            types[o] = obj_type
            added += 1

    print(f"Dodano {added} novih tipova za objekte.")
    return types

# safety check:
# 1. backup postojece datoteke ako postoji
# 2. ako je nova verzija prazna ili manja od stare (min_ratio) -> aborta
# 3. ako je sve ok -> mergea
def safe_save_entity_types(new_types: dict, output_path: str, min_ratio: float = 0.5):
    old_types = load_json(output_path)
    old_count = len(old_types)
    new_count = len(new_types)

    # ako vec postoji taj file, radi se njegov backup s timestampom
    if os.path.exists(output_path):
        backup_path = output_path.replace(
            ".json", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        shutil.copy2(output_path, backup_path)
        print(f"Backup napravljen: {backup_path}")

    # ako nema novih tipova
    if new_count == 0:
        print("ABORT: novi entity_types je prazan. Nista nije prepisano.")
        return False

    # ako je broj novih podataka puno manji od starih -> aborta jer je moguce da su izgubljeni neki podatci
    if old_count > 0 and new_count < old_count * min_ratio:
        print(
            f"ABORT: novi entity_types ({new_count}) je puno manji od postojeceg, mozda je doslo do greske."
        )
        return False
 
    # merge: stari podaci ostaju osim ako novi eksplicitno donosi vrijednost za isti kljuc
    merged = dict(old_types)
    merged.update(new_types)
 
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
 
    print(f"Spremljeno {len(merged)} entity_types u {output_path} (staro: {old_count}, novo dodano/spojeno: {new_count}).")
    return True

###############################################################################################################################

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "processed")
    base_dir = os.path.abspath(base_dir)
 
    triples_path = os.path.join(base_dir, "croatian_triples.txt")
    types_path = os.path.join(base_dir, "entity_types.json")
 
    existing_types = load_json(types_path)
    triples = load_triples(triples_path)
 
    print(f"Ucitano {len(existing_types)} postojecih tipova, {len(triples)} trojki.")
 
    new_types = update_entity_types(existing_types, triples)
    safe_save_entity_types(new_types, types_path)
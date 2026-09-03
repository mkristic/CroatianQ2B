"""
Poziv na Wikidata ne uspije uvijek vratiti sve entitete - javlja gresku 429 (too many requests).
U ovom fileu ucitavaju se trojke koje su uspjesno dohvacene, ponovno se pokusavaju dohvatiti kategorije koje NISU dohvacene te se rezultat sprema u isti dokument s vec dohvacenim.
"""

import os
import time
from croatian_kg_processor import CroatianKGProcessor

processor = CroatianKGProcessor()

# ucitavanje trojki koje su uspjesno dohvacene od prije
base_dir = os.path.dirname(os.path.abspath(__file__))
triples_path = os.path.join(base_dir, "..", "..", "data", "processed", "croatian_triples.txt")

with open(triples_path, encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")

        if len(parts) == 3:
            s, p, o = parts
            processor.triples.append((s, p, o))
            processor.entities.add(s)
            processor.entities.add(o)
            processor.relations.add(p)

print(f"Ucitano {len(processor.triples)} postojecih trojki iz prethodnog izvlacenja podataka")

# ponavljanje SAMO kategorija koje nisu dohvacene
failed_categories = [    
    processor.extract_croatian_landmarks,
    processor.extract_croatian_music_and_performing_arts,
]

for extract_fn in failed_categories:
    try:
        extract_fn(limit=50)
    except Exception as e:
        processor.logger.warning(f"Izvlacenje podataka ponovno nije uspjelo: {e}")
    time.sleep(30)

processor.clean_empty_entries()
processor.create_mappings()
processor.save_to_files()

stats = processor.get_statistics()
print("\n=== STATISTIKE NAKON RETRY-a ===")
for key, value in stats.items():
    if isinstance(value, list) and len(value) > 3:
        print(f"{key}: {value[:3]}...")
    else:
        print(f"{key}: {value}")
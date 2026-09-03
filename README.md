# Query2Box: Zaključivanje nad grafovima znanja u vektorskom prostoru pomoću ugnježđivanja u obliku kvadra

Prilagodba originalnog Query2Box modela sa Stanforda za graf znanja s hrvatskim entitetima izgrađen pomoću podataka iz Wikidata.\
Originalni rad:
- Ren, Hu, Leskovec — *Reasoning over Knowledge Graphs in Vector Space using Box Embeddings*
- GitHub poveznica: [Query2box](https://github.com/hyren/query2box)

--- 

### Cilj projekta
Po uzoru na spomenuti izvorni Query2Box projekt željelo se napraviti inačicu prilagođenu za hrvatski jezik.
Primijenjuje se *box-embedding* model koji odgovara na logičke upite nad grafom znanja koji je izgrađen na podatcima o Hrvatskoj iz baze podataka Wikidata. Primjeri dohvaćenih podataka: gradovi, osobe, kulturna baština, rijeke, ustanove itd.\
*Box-embedding* je geometrijski prikaz podataka u višedimenzionalnom prostoru, umjesto pomoću točaka u vektroskom prostoru. Takav prikaz omogućuje da se operacija presjeka više skupova modelira kao stvarni geometrijski presjek hiper-pravokutnika. Rezultati logičkih upita time postaju precizniji u odnosu na to kako bi bili prikazani da se radi u 2D prostoru.

#### Logički upiti u ovom modelu:
| Tip | Značenje | Primjer |
| --- | --- | --- |
| 1p | jedna projekcija (1 hop) | Koja je površina Hvara? |
| 2p | dvije projekcije (2 hop) | U kojoj se državi nalazi grad u kojemu je rođena osoba X? |
| 2i | presjek dvaju uvjeta (2i) | Koja je osoba rođena u Zagrebu i ima zanimanje odvjetnik? |

Radi vizualizacije upita i rezultata, izrađeno je Gradio sučelje koje uspoređuje odgovor dobiven pretragom grafa znanja s predikcijama treniranog modela.

---

### Arhitektura
1. Ekstrakcija RDF podataka s Wikidata
2. Pretvorba u format trojki (subject, predicate, object)
3. Generiranje skupa logičkih upita (1p, 2p, 2i)
4. Priprema dataseta 
5. Treniranje i evaluacija
6. Vizualizacija na sučelju

---

### Tehnologije
| Kategorija | Tehnologija |
| --- | --- |
| Dohvaćanje podataka | Wikidata Query Service (SPARQL endpoint), Python biblioteka SPARQLWrapper |
| Obrada podataka | Python biblioteke pandas, rdflib |
| Treniranje | PyTorch |
| Sučelje | Gradio |
| Jezik | Python 3.14 |

Cjelovit popis: [`requirements.txt`](./requirements.txt).

---

### Instalacija 
```bash
# kloniranje repozitorija i pozicioniranje u root
cd croatian_query2box

# stvaranje virtualnog okruženja
python -m venv query2box_env

# aktivacija
query2box_env\Scripts\activate.bat

# instalacija dependencyja
pip install -r requirements.txt
```
---

### Pokretanje
Pokretanje ide redom po etapama jer svaka nova etapa rabi izlaz prethodne.

#### Etapa 1 — ekstrakcija podataka

```bash
python src/data_processing/croatian_kg_processor.py
```

Dohvaća 13 kategorija hrvatskih entiteta s Wikidate (gradovi, osobe, znamenitosti, sveučilišta, instituti, festivali, UNESCO baština, knjige, glazba/scenska umjetnost, nacionalni parkovi, otoci, rijeke, planine) i sprema RDF trojke, ID mapiranja i tipove entiteta u `data/processed/`.

> **Napomena:** Wikidata Query Service povremeno vraća 429/502/504 greške zbog vlastite infrastrukturne nestabilnosti (dodatno objašnjeno na kraju). Ako neka kategorija ne uspije, pokrenuti:
> ```bash
> python src/data_processing/retry_failed_categories.py
> ```
> koji ponovno pokušava samo neuspjele kategorije, bez ponavljanja cijelog pipelinea.

#### Etapa 2 — generiranje logičkih upita

```bash
python src/data_processing/logical_query_generator.py
```

Generira do 300 upita svakog tipa (1p, 2p, 2i) i sprema ih u `data/queries/logical_queries.pkl` (služi za daljnji rad, datoteka nije čitljiva) i `logical_queries_sample.json` (čitljiv uzorak za uvid).

#### Etapa 3 — priprema dataseta

```bash
python src/models/dataset_split.py
```

Dijeli upite svakog tipa na train/valid/test (80/10/10) i dodaje negative sampleove potrebne za trening.

#### Etapa 4 — trening i evaluacija modela

```bash
python src/models/train_q2b.py
python src/models/evaluate.py
```

Trening sprema model u `results/models/box_embedding.pt`. Evaluacija ispisuje MRR i Hits@{1,3,10} po tipu upita, odvojeno za valid i test skup.

#### Etapa 5 — Gradio sučelje

```bash
python src/interface/app_v2.py
```

Otvara lokalno web sučelje gdje se za zadani upit prikazuje i točan odgovor iz grafa i top-5 predikcija treniranog modela, jedno pored drugog.\
(app.py je prva inačica aplikacije koja je nepotpuna i koja je služila kao pomoćni alat tijekom izrade projekta)

---

### Ograničenja
- Nestabilnost Wikidata Query Servicea (WDQS) - Wikidata prolazi migraciju backenda te zbog toga pozivi na bazu mogu biti neuspješni - WDQS vraća HTTP greške 429, 502 i 509. Zbog toga 2 od 13 planiranih entiteta nisu dohvaćeni. \
 Izvor: [Wikidata:SPARQL query service/WDQS backend update](https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/WDQS_backend_update).
 - Relativno mali graf (578 trojki, 551 entitet, 26 relacija) ograničava broj prirodno postojećih višehop lanaca i presjeka, što se posebno odražava na malen broj 2p primjera.






    

"""
Croatian Knowledge Graph Processor - izvor: Wikidata
Dohvaca hrvatske podatke na hrvatskom jeziku iz Wikidata
"""

import requests # slanje HTTP requestova 
import json # konvertiranje python -> JSON
from SPARQLWrapper import SPARQLWrapper, JSON # za queryje u RDF obliku
import pandas as pd
from rdflib import Graph, Namespace # za trojke (subjekt, relacija, objekt)
from typing import List, Tuple, Set, Dict
import logging
import time
import unicodedata
import os

class CroatianKGProcessor:
    def __init__(self):
        self.triples = []
        self.entities = set()
        self.relations = set()
        self.entity2id = {}
        self.relation2id = {}
        self.id2entity = {}
        self.id2relation = {}
        self.croatian_labels = {}  # cuva hrvatske prijevode
        
        # setup logging - za pracenje tijeka programa i errora
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    ###############################################################################################################################
    
    ### stvaranje RDF trojki za gradove, osobe i znamenitosti ####
    # RDF: (subject, predicate, object) -> (entity, property/relation, entity) -> (entitet, svojstvo, vrijednost)
    
    ########
    # izvlacenje -gradova- s hrvatskim nazivima i pretvaranje u format trojki 
    def extract_croatian_cities(self, limit: int = 500) -> List[Tuple[str, str, str]]:
        self.logger.info(f"Izvlacim hrvatske gradove iz Wikidata (limit: {limit})")
        
        # endpoint je Wikidata
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        
        # SPARQL upit koji vraca hrvatske gradove, neka njihova svojstva te njihove nazive i vrijednosti (na hrvatskom ako postoje)
        query = f"""
        SELECT DISTINCT ?grad ?gradLabel ?svojstvo ?vrijednost ?vrijednostLabel WHERE {{
            ?grad wdt:P31/wdt:P279* wd:Q515 .      # dohvaca sve entitete koji su gradovi
            ?grad wdt:P17 wd:Q224 .                # u Hrvatskoj
            ?grad ?svojstvo ?vrijednost .          # dohvaca svojstva i pripadne vrijednosti za svaki grad
            ?grad rdfs:label ?gradLabel .          # rdfs:label umjesto Wikidata ID-ja (npr. wd:Q1435) daje citljivo ime (npr. Zagreb)
            FILTER(LANG(?gradLabel) = "hr")        # # dohvaca se hrvatski naziv grada
            
            # zelimo samo neka svojstva
            VALUES ?svojstvo {{
                wdt:P17      # drzava
                wdt:P131     # nalazi se u
                wdt:P625     # koordinate
                wdt:P1082    # broj stanovnika
                wdt:P571     # datum osnivanja
                wdt:P281     # postanski broj
                wdt:P37      # sluzbeni jezik
            }}
            
            # trazimo naziv vrijednosti za entitete na hrvatskom, optional ne vraca gresku ako ne pronade nista 
            OPTIONAL {{                                         
                ?vrijednost rdfs:label ?vrijednostLabel .
                FILTER(LANG(?vrijednostLabel) = "hr")      
            }}
            
            # uz label rezultata, dobije se i vrijednost (npr. wd:Q1435 Zagreb, umjesto samo wd:Q1435)
            SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "hr,en" .
            }}
        }}
        LIMIT {limit}
        """
        
        # spremanje upita u JSON
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        
        # ako dode do greske, except vraca []
        try:
            results = sparql.query().convert()
            
            # dohvacanje vrijednosti varijabli
            for result in results["results"]["bindings"]:
                grad = result["gradLabel"]["value"] if "gradLabel" in result else result["grad"]["value"]
                svojstvo = self.translate_property(result["svojstvo"]["value"])
                vrijednost = result["vrijednostLabel"]["value"] if "vrijednostLabel" in result else result["vrijednost"]["value"]
                
                # pojednostavljivanje naziva
                grad = self.clean_name(grad)
                vrijednost = self.clean_name(vrijednost) if not vrijednost.isdigit() else vrijednost
                
                # stvaranje RDF trojki
                self.triples.append((grad, svojstvo, vrijednost))
                self.entities.add(grad)
                self.entities.add(vrijednost)
                self.relations.add(svojstvo)
                
            self.logger.info(f"Izvuceno {len(self.triples)} trojki o hrvatskim gradovima")
            del results # oslobadanje memorije (SPARQL upiti mogu vratiti velike kolicine podataka, eksplicitnim brisanjem brze oslobodimo memoriju)
            return self.triples
            
        except Exception as e:
            self.logger.error(f"Greska pri izvlacenju gradova: {e}")
            return []
    
    ########
    # izvlacenje -osoba- s hrvatskim nazivima i pretvaranje u format trojki 
    def extract_croatian_people(self, limit: int = 300) -> List[Tuple[str, str, str]]:
        self.logger.info(f"Izvlacim hrvatske osobe iz Wikidata (limit: {limit})")
        
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        
        query = f"""
        SELECT DISTINCT ?osoba ?osobaLabel ?svojstvo ?vrijednost ?vrijednostLabel WHERE {{
            ?osoba wdt:P27 wd:Q224 .               # hrvatsko drzavljanstvo
            ?osoba wdt:P31 wd:Q5 .                 # je osoba
            ?osoba ?svojstvo ?vrijednost .
            ?osoba rdfs:label ?osobaLabel .
            FILTER(LANG(?osobaLabel) = "hr")
            
            VALUES ?svojstvo {{
                wdt:P27      # drzavljanstvo
                wdt:P19      # mjesto rodenja
                wdt:P106     # zanimanje
                wdt:P569     # datum rodenja
                wdt:P20      # mjesto smrti
                wdt:P26      # bracni partner
                wdt:P69      # obrazovanje
            }}
            
            OPTIONAL {{
                ?vrijednost rdfs:label ?vrijednostLabel .
                FILTER(LANG(?vrijednostLabel) = "hr")
            }}
            
            SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "hr,en" .
            }}
        }}
        LIMIT {limit}
        """
        
        # spremanje upita u JSON
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        
        # ako dode do greske, except vraca []
        try:
            # Wikidata ima limit koliko upita u sekundi moze primiti
            # time.sleep() napravi pauzu da se Wikidata ne preoptereti upitima i da ne dode do gresaka/crashanja
            time.sleep(1)  
            results = sparql.query().convert()
            
            # dohvacanje vrijednosti varijabli
            for result in results["results"]["bindings"]:
                osoba = result["osobaLabel"]["value"] if "osobaLabel" in result else result["osoba"]["value"]
                svojstvo = self.translate_property(result["svojstvo"]["value"])
                vrijednost = result["vrijednostLabel"]["value"] if "vrijednostLabel" in result else result["vrijednost"]["value"]
                
                # pojednostavljivanje naziva
                osoba = self.clean_name(osoba)
                vrijednost = self.clean_name(vrijednost)
                
                # stvaranje RDF trojki
                self.triples.append((osoba, svojstvo, vrijednost))
                self.entities.add(osoba)
                self.entities.add(vrijednost)
                self.relations.add(svojstvo)
                
            self.logger.info(f"Izvuceno {len(self.triples)} trojki o hrvatskim osobama")
            del results # oslobadanje memorije (SPARQL upiti mogu vratiti velike kolicine podataka, eksplicitnim brisanjem brze oslobodimo memoriju)
            return self.triples
        
        except Exception as e:
            self.logger.error(f"Greska pri izvlacenju osoba: {e}")
            return []
    
    ########
    # izvlacenje -znamenitosti- s hrvatskim nazivima i pretvaranje u format trojki 
    def extract_croatian_landmarks(self, limit: int = 200) -> List[Tuple[str, str, str]]:
        self.logger.info(f"Izvlacim hrvatske znamenitosti iz Wikidata (limit: {limit})")
        
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        
        query = f"""
        SELECT DISTINCT ?znamenitost ?znamenitostLabel ?svojstvo ?vrijednost ?vrijednostLabel WHERE {{
            {{
                ?znamenitost wdt:P17 wd:Q224 .                  # u Hrvatskoj
                ?znamenitost wdt:P31/wdt:P279* wd:Q570116 .     # kulturna bastina
            }} UNION {{
                ?znamenitost wdt:P17 wd:Q224 .
                ?znamenitost wdt:P31/wdt:P279* wd:Q4989906 .    # spomenik
            }} UNION {{
                ?znamenitost wdt:P17 wd:Q224 .
                ?znamenitost wdt:P31/wdt:P279* wd:Q33506 .      # muzej
            }}
            
            ?znamenitost ?svojstvo ?vrijednost .
            ?znamenitost rdfs:label ?znamenitostLabel .
            FILTER(LANG(?znamenitostLabel) = "hr")
            
            VALUES ?svojstvo {{
                wdt:P17      # drzava
                wdt:P131     # nalazi se u
                wdt:P571     # osnovan
                wdt:P84      # arhitekt
                wdt:P149     # arhitektonski stil
            }}
            
            OPTIONAL {{
                ?vrijednost rdfs:label ?vrijednostLabel .
                FILTER(LANG(?vrijednostLabel) = "hr")
            }}
            
            SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "hr,en" .
            }}
        }}
        LIMIT {limit}
        """
        
        # spremanje upita u JSON
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        
        try:
            # Wikidata ima limit koliko upita u sekundi moze primiti
            # time.sleep() napravi pauzu da se Wikidata ne preoptereti upitima i da ne dode do gresaka/crashanja
            time.sleep(1)
            results = sparql.query().convert()
            
            # dohvacanje vrijednosti varijabli
            for result in results["results"]["bindings"]:
                znamenitost = result["znamenitostLabel"]["value"] if "znamenitostLabel" in result else result["znamenitost"]["value"]
                svojstvo = self.translate_property(result["svojstvo"]["value"]) 
                vrijednost = result["vrijednostLabel"]["value"] if "vrijednostLabel" in result else result["vrijednost"]["value"]
                
                # pojednostavljivanje naziva
                znamenitost = self.clean_name(znamenitost)
                vrijednost = self.clean_name(vrijednost)
                
                # stvaranje RDF trojki
                self.triples.append((znamenitost, svojstvo, vrijednost))
                self.entities.add(znamenitost)
                self.entities.add(vrijednost)
                self.relations.add(svojstvo)
                
            self.logger.info(f"Izvuceno {len(self.triples)} trojki o hrvatskim znamenitostima")
            del results # oslobadanje memorije (SPARQL upiti mogu vratiti velike kolicine podataka, eksplicitnim brisanjem brze oslobodimo memoriju)
            return self.triples
        
        except Exception as e:
            self.logger.error(f"Greska pri izvlacenju znamenitosti: {e}")
            return []

    ########
    # izvlacenje -sveucilista- s hrvatskim nazivima i pretvaranje u format trojki
    def extract_croatian_universities(self, limit: int = 100) -> List[Tuple[str, str, str]]:
        self.logger.info(f"Izvlacim hrvatska sveucilista iz Wikidata (limit: {limit})")

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

        query = f""" 
        SELECT DISTINCT ?sveuciliste ?sveucilisteLabel ?svojstvo ?vrijednost ?vrijednostLabel WHERE {{
            ?sveuciliste wdt:P31/wdt:P279* wd:Q3918 .
            ?sveuciliste wdt:P17 wd:Q224 .
            ?sveuciliste ?svojstvo ?vrijednost .
            ?sveuciliste rdfs:label ?sveucilisteLabel .
            FILTER(LANG(?sveucilisteLabel) = "hr")

            # zelimo samo neka svojstva
            VALUES ?svojstvo {{
                wdt:P17 	# drzava
                wdt:P131 	# nalazi se u
                wdt:P571 	# datum osnivanja
                wdt:P856 	# sluzbena stranica
                wdt:P1448 	# sluzbeni naziv
                wdt:P159    # sjediste
            }}

            OPTIONAL {{                                         
                ?vrijednost rdfs:label ?vrijednostLabel .
                FILTER(LANG(?vrijednostLabel) = "hr")      
            }}
            
            SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "hr,en" .
            }}
        }}
        LIMIT {limit}
        """

        # spremanje upita u JSON
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        
        # ako dode do greske, except vraca []
        try:
            time.sleep(1)
            results = sparql.query().convert()
            
            # dohvacanje vrijednosti varijabli
            for result in results["results"]["bindings"]:
                sveuciliste = result["sveucilisteLabel"]["value"] if "sveucilisteLabel" in result else result["sveuciliste"]["value"]
                svojstvo = self.translate_property(result["svojstvo"]["value"])
                vrijednost = result["vrijednostLabel"]["value"] if "vrijednostLabel" in result else result["vrijednost"]["value"]
                
                # pojednostavljivanje naziva
                sveuciliste = self.clean_name(sveuciliste)
                vrijednost = self.clean_name(vrijednost) if not vrijednost.isdigit() else vrijednost
                
                # stvaranje RDF trojki
                self.triples.append((sveuciliste, svojstvo, vrijednost))
                self.entities.add(sveuciliste)
                self.entities.add(vrijednost)
                self.relations.add(svojstvo)
                
            self.logger.info(f"Izvuceno {len(self.triples)} trojki o hrvatskim sveucilistima")
            del results # oslobadanje memorije (SPARQL upiti mogu vratiti velike kolicine podataka, eksplicitnim brisanjem brze oslobodimo memoriju)
            return self.triples
            
        except Exception as e:
            self.logger.error(f"Greska pri izvlacenju sveucilista: {e}")
            return []

    ########    
    # izvlacenje -instituta- s hrvatskim nazivima i pretvaranje u format trojki
    def extract_croatian_institutes(self, limit: int = 100) -> List[Tuple[str, str, str]]:
        self.logger.info(f"Izvlacim hrvatske institute iz Wikidata (limit: {limit})")

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

        query = f""" 
        SELECT DISTINCT ?institut ?institutLabel ?svojstvo ?vrijednost ?vrijednostLabel WHERE {{
            ?institut wdt:P31/wdt:P279* wd:Q31855 .
            ?institut wdt:P17 wd:Q224 .
            ?institut ?svojstvo ?vrijednost .
            ?institut rdfs:label ?institutLabel .
            FILTER(LANG(?institutLabel) = "hr")

            # zelimo samo neka svojstva
            VALUES ?svojstvo {{
                wdt:P17 	# drzava
                wdt:P131 	# nalazi se u
                wdt:P571 	# datum osnivanja
                wdt:P856 	# sluzbena stranica
                wdt:P1037 	# ravnatelj
                wdt:P101    # podrucje rada
            }}

            OPTIONAL {{                                         
                ?vrijednost rdfs:label ?vrijednostLabel .
                FILTER(LANG(?vrijednostLabel) = "hr")      
            }}
            
            SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "hr,en" .
            }}
        }}
        LIMIT {limit}
        """

        # spremanje upita u JSON
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        
        # ako dode do greske, except vraca []
        try:
            time.sleep(1)
            results = sparql.query().convert()
            
            # dohvacanje vrijednosti varijabli
            for result in results["results"]["bindings"]:
                institut = result["institutLabel"]["value"] if "institutLabel" in result else result["institut"]["value"]
                svojstvo = self.translate_property(result["svojstvo"]["value"])
                vrijednost = result["vrijednostLabel"]["value"] if "vrijednostLabel" in result else result["vrijednost"]["value"]
                
                # pojednostavljivanje naziva
                institut = self.clean_name(institut)
                vrijednost = self.clean_name(vrijednost) if not vrijednost.isdigit() else vrijednost
                
                # stvaranje RDF trojki
                self.triples.append((institut, svojstvo, vrijednost))
                self.entities.add(institut)
                self.entities.add(vrijednost)
                self.relations.add(svojstvo)
                
            self.logger.info(f"Izvuceno {len(self.triples)} trojki o hrvatskim institutima")
            del results # oslobadanje memorije (SPARQL upiti mogu vratiti velike kolicine podataka, eksplicitnim brisanjem brze oslobodimo memoriju)
            return self.triples
            
        except Exception as e:
            self.logger.error(f"Greska pri izvlacenju instituta: {e}")
            return []
    
    ########
    # izvlacenje -festivala- s hrvatskim nazivima i pretvaranje u format trojki
    def extract_croatian_festivals(self, limit: int = 100) -> List[Tuple[str, str, str]]:
        self.logger.info(f"Izvlacim hrvatske festivale iz Wikidata (limit: {limit})")

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

        query = f""" 
        SELECT DISTINCT ?festival ?festivalLabel ?svojstvo ?vrijednost ?vrijednostLabel WHERE {{
            ?festival wdt:P31 wd:Q132241 .
            ?festival wdt:P17 wd:Q224 .
            ?festival ?svojstvo ?vrijednost .
            ?festival rdfs:label ?festivalLabel .
            FILTER(LANG(?festivalLabel) = "hr")

            # zelimo samo neka svojstva
            VALUES ?svojstvo {{
                wdt:P17 	# drzava
                wdt:P131 	# nalazi se u
                wdt:P571 	# datum osnivanja
                wdt:P859    # sponzor
                wdt:P664    # organizator
                wdt:P710    # sudionik
            }}

            OPTIONAL {{                                         
                ?vrijednost rdfs:label ?vrijednostLabel .
                FILTER(LANG(?vrijednostLabel) = "hr")      
            }}
            
            SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "hr,en" .
            }}
        }}
        LIMIT {limit}
        """

        # spremanje upita u JSON
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        
        # ako dode do greske, except vraca []
        try:
            time.sleep(1)
            results = sparql.query().convert()
            
            # dohvacanje vrijednosti varijabli
            for result in results["results"]["bindings"]:
                festival = result["festivalLabel"]["value"] if "festivalLabel" in result else result["festival"]["value"]
                svojstvo = self.translate_property(result["svojstvo"]["value"])
                vrijednost = result["vrijednostLabel"]["value"] if "vrijednostLabel" in result else result["vrijednost"]["value"]
                
                # pojednostavljivanje naziva
                festival = self.clean_name(festival)
                vrijednost = self.clean_name(vrijednost) if not vrijednost.isdigit() else vrijednost
                
                # stvaranje RDF trojki
                self.triples.append((festival, svojstvo, vrijednost))
                self.entities.add(festival)
                self.entities.add(vrijednost)
                self.relations.add(svojstvo)
                
            self.logger.info(f"Izvuceno {len(self.triples)} trojki o hrvatskim festivalima")
            del results # oslobadanje memorije (SPARQL upiti mogu vratiti velike kolicine podataka, eksplicitnim brisanjem brze oslobodimo memoriju)
            return self.triples
            
        except Exception as e:
            self.logger.error(f"Greska pri izvlacenju festivala: {e}")
            return []
    
    ########
    # izvlacenje -UNESCO lokaliteta svjetske bastine- s hrvatskim nazivima i pretvaranje u format trojki
    def extract_croatian_world_heritage_sites(self, limit: int = 100) -> List[Tuple[str, str, str]]:
        self.logger.info(f"Izvlacim hrvatske lokalitete na UNESCO-vom popisu svjetske bastine iz Wikidata (limit: {limit})")

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

        query = f""" 
        SELECT DISTINCT ?bastina ?bastinaLabel ?svojstvo ?vrijednost ?vrijednostLabel WHERE {{
            ?bastina wdt:P31 wd:Q9259 .
            ?bastina wdt:P17 wd:Q224 .
            ?bastina ?svojstvo ?vrijednost .
            ?bastina rdfs:label ?bastinaLabel .
            FILTER(LANG(?bastinaLabel) = "hr")

            # zelimo samo neka svojstva
            VALUES ?svojstvo {{
                wdt:P17 	# drzava
                wdt:P131 	# nalazi se u       
                wdt:P571 	# datum osnivanja         
                wdt:P856    # sluzbena stranica
                wdt:P527    # sastoji se od
                wdt:P1174   # godisnji broj posjetitelja
            }}

            OPTIONAL {{                                         
                ?vrijednost rdfs:label ?vrijednostLabel .
                FILTER(LANG(?vrijednostLabel) = "hr")      
            }}
            
            SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "hr,en" .
            }}
        }}
        LIMIT {limit}
        """

        # spremanje upita u JSON
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        
        # ako dode do greske, except vraca []
        try:
            time.sleep(1)
            results = sparql.query().convert()
            
            # dohvacanje vrijednosti varijabli
            for result in results["results"]["bindings"]:
                bastina = result["bastinaLabel"]["value"] if "bastinaLabel" in result else result["bastina"]["value"]
                svojstvo = self.translate_property(result["svojstvo"]["value"])
                vrijednost = result["vrijednostLabel"]["value"] if "vrijednostLabel" in result else result["vrijednost"]["value"]
                
                # pojednostavljivanje naziva
                bastina = self.clean_name(bastina)
                vrijednost = self.clean_name(vrijednost) if not vrijednost.isdigit() else vrijednost
                
                # stvaranje RDF trojki
                self.triples.append((bastina, svojstvo, vrijednost))
                self.entities.add(bastina)
                self.entities.add(vrijednost)
                self.relations.add(svojstvo)
                
            self.logger.info(f"Izvuceno {len(self.triples)} trojki o hrvatskim lokalitetima na UNESCO-vom popisu svjetske bastine")
            del results # oslobadanje memorije (SPARQL upiti mogu vratiti velike kolicine podataka, eksplicitnim brisanjem brze oslobodimo memoriju)
            return self.triples
            
        except Exception as e:
            self.logger.error(f"Greska pri izvlacenju hrvatskih lokaliteta na UNESCO-vom popisu svjetske bastine: {e}")
            return []

    #######
    # izvlacenje -knjizevnih djela- s hrvatskim nazivima i pretvaranje u format trojki
    def extract_croatian_books(self, limit: int = 300) -> List[Tuple[str, str, str]]:
        self.logger.info(f"Izvlacim hrvatska knjizevna djela iz Wikidata (limit: {limit})")

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

        query = f""" 
        SELECT DISTINCT ?knjiga ?knjigaLabel ?svojstvo ?vrijednost ?vrijednostLabel WHERE {{
            ?knjiga wdt:P31/wdt:P279* wd:Q7725634 .
            ?knjiga wdt:P17 wd:Q224 .
            ?knjiga ?svojstvo ?vrijednost .
            ?knjiga rdfs:label ?knjigaLabel .
            FILTER(LANG(?knjigaLabel) = "hr")

            # zelimo samo neka svojstva
            VALUES ?svojstvo {{
                wdt:P50     # pisac
                wdt:P136    # zanr
                wdt:P921    # glavna tema
                wdt:P577    # datum izdavanja
                wdt:P166    # nagrada
                wdt:P123    # izdavac
            }}

            OPTIONAL {{                                         
                ?vrijednost rdfs:label ?vrijednostLabel .
                FILTER(LANG(?vrijednostLabel) = "hr")      
            }}
            
            SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "hr,en" .
            }}
        }}
        LIMIT {limit}
        """

        # spremanje upita u JSON
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        
        # ako dode do greske, except vraca []
        try:
            time.sleep(1)
            results = sparql.query().convert()
            
            # dohvacanje vrijednosti varijabli
            for result in results["results"]["bindings"]:
                knjiga = result["knjigaLabel"]["value"] if "knjigaLabel" in result else result["knjiga"]["value"]
                svojstvo = self.translate_property(result["svojstvo"]["value"])
                vrijednost = result["vrijednostLabel"]["value"] if "vrijednostLabel" in result else result["vrijednost"]["value"]
                
                # pojednostavljivanje naziva
                knjiga = self.clean_name(knjiga)
                vrijednost = self.clean_name(vrijednost) if not vrijednost.isdigit() else vrijednost
                
                # stvaranje RDF trojki
                self.triples.append((knjiga, svojstvo, vrijednost))
                self.entities.add(knjiga)
                self.entities.add(vrijednost)
                self.relations.add(svojstvo)
                
            self.logger.info(f"Izvuceno {len(self.triples)} trojki o hrvatskim knjizevnim djelima")
            del results # oslobadanje memorije (SPARQL upiti mogu vratiti velike kolicine podataka, eksplicitnim brisanjem brze oslobodimo memoriju)
            return self.triples
            
        except Exception as e:
            self.logger.error(f"Greska pri izvlacenju knjizevnih djela: {e}")
            return []

    #######
    # izvlacenje -glazbene i scenske umjetnosti- s hrvatskim nazivima i pretvaranje u format trojki 
    def extract_croatian_music_and_performing_arts(self, limit: int = 100) -> List[Tuple[str, str, str]]:
        self.logger.info(f"Izvlacim hrvatsku glazbenu i scensku umjetnost iz Wikidata (limit: {limit})")
        
        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        
        query = f"""
        SELECT DISTINCT ?umjetnost ?umjetnostLabel ?svojstvo ?vrijednost ?vrijednostLabel WHERE {{
            {{
                ?umjetnost wdt:P17 wd:Q224 .                    # u Hrvatskoj
                ?umjetnost wdt:P31/wdt:P279* wd:Q482994 .       # glazbeni album
            }} UNION {{
                ?umjetnost wdt:P17 wd:Q224 .
                ?umjetnost wdt:P31/wdt:P279* wd:Q2188189 .      # glazbeno djelo (obuhvaca pjesme, skladbe, albume...)
            }} UNION {{
                ?umjetnost wdt:P17 wd:Q224 .
                ?umjetnost wdt:P31/wdt:P279* wd:Q215380 .       # glazbena grupa
            }} UNION {{
                ?umjetnost wdt:P17 wd:Q224 .
                ?umjetnost wdt:P31/wdt:P279* wd:Q34379 .        # glazbeni instrument
            }} UNION {{
                ?umjetnost wdt:P17 wd:Q224 .
                ?umjetnost wdt:P31/wdt:P279* wd:Q639669 .       # glazbenik
            }} UNION {{
                ?umjetnost wdt:P17 wd:Q224 .
                ?umjetnost wdt:P31/wdt:P279* wd:Q177220 .       # pjevac
            }} UNION {{
                ?umjetnost wdt:P17 wd:Q224 .
                ?umjetnost wdt:P31/wdt:P279* wd:Q868557 .       # glazbeni festival
            }} UNION {{
                ?umjetnost wdt:P17 wd:Q224 .
                ?umjetnost wdt:P31/wdt:P279* wd:Q41425 .        # balet
            }} UNION {{
                ?umjetnost wdt:P17 wd:Q224 .
                ?umjetnost wdt:P31/wdt:P279* wd:Q4070300 .      # balerina
            }} UNION {{
                ?umjetnost wdt:P17 wd:Q224 .
                ?umjetnost wdt:P31/wdt:P279* wd:Q25379 .        # predstava
            }} UNION {{
                ?umjetnost wdt:P17 wd:Q224 .
                ?umjetnost wdt:P31/wdt:P279* wd:Q24354 .        # kazaliste
            }}
            
            ?umjetnost ?svojstvo ?vrijednost .
            ?umjetnost rdfs:label ?umjetnostLabel .
            FILTER(LANG(?umjetnostLabel) = "hr")
            
            VALUES ?svojstvo {{
                wdt:P175      # izvodac
                wdt:P571      # datum osnivanja
                wdt:P136      # zanr
                wdt:P1476     # naziv
                wdt:P2047     # trajanje
                wdt:P162      # producent
                wdt:P86       # skladatelj
                wdt:P166      # nagrada
                wdt:P1303     # (svira) glazbeni instrument
            }}
            
            OPTIONAL {{
                ?vrijednost rdfs:label ?vrijednostLabel .
                FILTER(LANG(?vrijednostLabel) = "hr")
            }}
            
            SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "hr,en" .
            }}
        }}
        LIMIT {limit}
        """
        
        # spremanje upita u JSON
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        
        try:
            # Wikidata ima limit koliko upita u sekundi moze primiti
            # time.sleep() napravi pauzu da se Wikidata ne preoptereti upitima i da ne dode do gresaka/crashanja
            time.sleep(1)
            results = sparql.query().convert()
            
            # dohvacanje vrijednosti varijabli
            for result in results["results"]["bindings"]:
                umjetnost = result["umjetnostLabel"]["value"] if "umjetnostLabel" in result else result["umjetnost"]["value"]
                svojstvo = self.translate_property(result["svojstvo"]["value"]) 
                vrijednost = result["vrijednostLabel"]["value"] if "vrijednostLabel" in result else result["vrijednost"]["value"]
                
                # pojednostavljivanje naziva
                umjetnost = self.clean_name(umjetnost)
                vrijednost = self.clean_name(vrijednost)
                
                # stvaranje RDF trojki
                self.triples.append((umjetnost, svojstvo, vrijednost))
                self.entities.add(umjetnost)
                self.entities.add(vrijednost)
                self.relations.add(svojstvo)
                
            self.logger.info(f"Izvuceno {len(self.triples)} trojki o hrvatskoj glazbenoj i scenskoj umjetnosti")
            del results # oslobadanje memorije (SPARQL upiti mogu vratiti velike kolicine podataka, eksplicitnim brisanjem brze oslobodimo memoriju)
            return self.triples
        
        except Exception as e:
            self.logger.error(f"Greska pri izvlacenju glazbene i scenske umjetnosti: {e}")
            return []
    
    #######
    # izvlacenje -nacionalnih parkova- s hrvatskim nazivima i pretvaranje u format trojki
    def extract_croatian_national_parks(self, limit: int = 100) -> List[Tuple[str, str, str]]:
        self.logger.info(f"Izvlacim hrvatske nacionalne parkove iz Wikidata (limit: {limit})")

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

        query = f""" 
        SELECT DISTINCT ?npark ?nparkLabel ?svojstvo ?vrijednost ?vrijednostLabel WHERE {{
            ?npark wdt:P31 wd:Q46169 .
            ?npark wdt:P17 wd:Q224 .
            ?npark ?svojstvo ?vrijednost .
            ?npark rdfs:label ?nparkLabel .
            FILTER(LANG(?nparkLabel) = "hr")

            # zelimo samo neka svojstva
            VALUES ?svojstvo {{
                wdt:P571        # datum osnivanja
                wdt:P856        # sluzbena stranica
                wdt:P1174       # godisnji broj posjetitelja
                wdt:P2046       # povrsina
                wdt:P610        # najvisa tocka
                wdt:P131        # nalazi se u
            }}

            OPTIONAL {{                                         
                ?vrijednost rdfs:label ?vrijednostLabel .
                FILTER(LANG(?vrijednostLabel) = "hr")      
            }}
            
            SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "hr,en" .
            }}
        }}
        LIMIT {limit}
        """

        # spremanje upita u JSON
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        
        # ako dode do greske, except vraca []
        try:
            time.sleep(1)
            results = sparql.query().convert()
            
            # dohvacanje vrijednosti varijabli
            for result in results["results"]["bindings"]:
                npark = result["nparkLabel"]["value"] if "nparkLabel" in result else result["npark"]["value"]
                svojstvo = self.translate_property(result["svojstvo"]["value"])
                vrijednost = result["vrijednostLabel"]["value"] if "vrijednostLabel" in result else result["vrijednost"]["value"]
                
                # pojednostavljivanje naziva
                npark = self.clean_name(npark)
                vrijednost = self.clean_name(vrijednost) if not vrijednost.isdigit() else vrijednost
                
                # stvaranje RDF trojki
                self.triples.append((npark, svojstvo, vrijednost))
                self.entities.add(npark)
                self.entities.add(vrijednost)
                self.relations.add(svojstvo)
                
            self.logger.info(f"Izvuceno {len(self.triples)} trojki o hrvatskim nacionalnim parkovima")
            del results # oslobadanje memorije (SPARQL upiti mogu vratiti velike kolicine podataka, eksplicitnim brisanjem brze oslobodimo memoriju)
            return self.triples
            
        except Exception as e:
            self.logger.error(f"Greska pri izvlacenju nacionalnih parkova: {e}")
            return []

    #######
    # izvlacenje -otoka- s hrvatskim nazivima i pretvaranje u format trojki
    def extract_croatian_islands(self, limit: int = 100) -> List[Tuple[str, str, str]]:
        self.logger.info(f"Izvlacim hrvatske otoke iz Wikidata (limit: {limit})")

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

        query = f""" 
        SELECT DISTINCT ?otok ?otokLabel ?svojstvo ?vrijednost ?vrijednostLabel WHERE {{
            ?otok wdt:P31/wdt:P279* wd:Q23442 .
            ?otok wdt:P17 wd:Q224 .
            ?otok ?svojstvo ?vrijednost .
            ?otok rdfs:label ?otokLabel .
            FILTER(LANG(?otokLabel) = "hr")

            # zelimo samo neka svojstva
            VALUES ?svojstvo {{
                wdt:P1082       # broj stanovnika
                wdt:P17         # drzava
                wdt:P131        # nalazi se u
                wdt:P2046       # povrsina
                wdt:P610        # najvisa tocka
            }}

            OPTIONAL {{                                         
                ?vrijednost rdfs:label ?vrijednostLabel .
                FILTER(LANG(?vrijednostLabel) = "hr")      
            }}
            
            SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "hr,en" .
            }}
        }}
        LIMIT {limit}
        """

        # spremanje upita u JSON
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        
        # ako dode do greske, except vraca []
        try:
            time.sleep(1)
            results = sparql.query().convert()
            
            # dohvacanje vrijednosti varijabli
            for result in results["results"]["bindings"]:
                otok = result["otokLabel"]["value"] if "otokLabel" in result else result["otok"]["value"]
                svojstvo = self.translate_property(result["svojstvo"]["value"])
                vrijednost = result["vrijednostLabel"]["value"] if "vrijednostLabel" in result else result["vrijednost"]["value"]
                
                # pojednostavljivanje naziva
                otok = self.clean_name(otok)
                vrijednost = self.clean_name(vrijednost) if not vrijednost.isdigit() else vrijednost
                
                # stvaranje RDF trojki
                self.triples.append((otok, svojstvo, vrijednost))
                self.entities.add(otok)
                self.entities.add(vrijednost)
                self.relations.add(svojstvo)
                
            self.logger.info(f"Izvuceno {len(self.triples)} trojki o hrvatskim otocima")
            del results # oslobadanje memorije (SPARQL upiti mogu vratiti velike kolicine podataka, eksplicitnim brisanjem brze oslobodimo memoriju)
            return self.triples
            
        except Exception as e:
            self.logger.error(f"Greska pri izvlacenju otoka: {e}")
            return []

    #######
    # izvlacenje -rijeka- s hrvatskim nazivima i pretvaranje u format trojki
    def extract_croatian_rivers(self, limit: int = 100) -> List[Tuple[str, str, str]]:
        self.logger.info(f"Izvlacim hrvatske rijeke iz Wikidata (limit: {limit})")

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

        query = f""" 
        SELECT DISTINCT ?rijeka ?rijekaLabel ?svojstvo ?vrijednost ?vrijednostLabel WHERE {{
            ?rijeka wdt:P31/wdt:P279* wd:Q4022 .
            ?rijeka wdt:P17 wd:Q224 .
            ?rijeka ?svojstvo ?vrijednost .
            ?rijeka rdfs:label ?rijekaLabel .
            FILTER(LANG(?rijekaLabel) = "hr")

            # zelimo samo neka svojstva
            VALUES ?svojstvo {{
                wdt:P885        # izvor
                wdt:P403        # usce
                wdt:P17         # drzava
            }}

            OPTIONAL {{                                         
                ?vrijednost rdfs:label ?vrijednostLabel .
                FILTER(LANG(?vrijednostLabel) = "hr")      
            }}
            
            SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "hr,en" .
            }}
        }}
        LIMIT {limit}
        """

        # spremanje upita u JSON
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        
        # ako dode do greske, except vraca []
        try:
            time.sleep(1)
            results = sparql.query().convert()
            
            # dohvacanje vrijednosti varijabli
            for result in results["results"]["bindings"]:
                rijeka = result["rijekaLabel"]["value"] if "rijekaLabel" in result else result["rijeka"]["value"]
                svojstvo = self.translate_property(result["svojstvo"]["value"])
                vrijednost = result["vrijednostLabel"]["value"] if "vrijednostLabel" in result else result["vrijednost"]["value"]
                
                # pojednostavljivanje naziva
                rijeka = self.clean_name(rijeka)
                vrijednost = self.clean_name(vrijednost) if not vrijednost.isdigit() else vrijednost
                
                # stvaranje RDF trojki
                self.triples.append((rijeka, svojstvo, vrijednost))
                self.entities.add(rijeka)
                self.entities.add(vrijednost)
                self.relations.add(svojstvo)
                
            self.logger.info(f"Izvuceno {len(self.triples)} trojki o hrvatskim rijekama")
            del results # oslobadanje memorije (SPARQL upiti mogu vratiti velike kolicine podataka, eksplicitnim brisanjem brze oslobodimo memoriju)
            return self.triples
            
        except Exception as e:
            self.logger.error(f"Greska pri izvlacenju rijeka: {e}")
            return []

    #######
    # izvlacenje -planina- s hrvatskim nazivima i pretvaranje u format trojki
    def extract_croatian_mountains(self, limit: int = 100) -> List[Tuple[str, str, str]]:
        self.logger.info(f"Izvlacim hrvatske planine iz Wikidata (limit: {limit})")

        sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

        query = f""" 
        SELECT DISTINCT ?planina ?planinaLabel ?svojstvo ?vrijednost ?vrijednostLabel WHERE {{
            ?planina wdt:P31/wdt:P279* wd:Q8502 .
            ?planina wdt:P17 wd:Q224 .
            ?planina ?svojstvo ?vrijednost .
            ?planina rdfs:label ?planinaLabel .
            FILTER(LANG(?planinaLabel) = "hr")

            # zelimo samo neka svojstva
            VALUES ?svojstvo {{
                wdt:P527        # sastoji se od
                wdt:P610        # najvisa tocka
                wdt:P17         # drzava
                wdt:P4552       # gorje
                wdt:P2044       # nadmorska visina
            }}

            OPTIONAL {{                                         
                ?vrijednost rdfs:label ?vrijednostLabel .
                FILTER(LANG(?vrijednostLabel) = "hr")      
            }}
            
            SERVICE wikibase:label {{ 
                bd:serviceParam wikibase:language "hr,en" .
            }}
        }}
        LIMIT {limit}
        """

        # spremanje upita u JSON
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        
        # ako dode do greske, except vraca []
        try:
            time.sleep(1)
            results = sparql.query().convert()
            
            # dohvacanje vrijednosti varijabli
            for result in results["results"]["bindings"]:
                planina = result["planinaLabel"]["value"] if "planinaLabel" in result else result["planina"]["value"]
                svojstvo = self.translate_property(result["svojstvo"]["value"])
                vrijednost = result["vrijednostLabel"]["value"] if "vrijednostLabel" in result else result["vrijednost"]["value"]
                
                # pojednostavljivanje naziva
                planina = self.clean_name(planina)
                vrijednost = self.clean_name(vrijednost) if not vrijednost.isdigit() else vrijednost
                
                # stvaranje RDF trojki
                self.triples.append((planina, svojstvo, vrijednost))
                self.entities.add(planina)
                self.entities.add(vrijednost)
                self.relations.add(svojstvo)
                
            self.logger.info(f"Izvuceno {len(self.triples)} trojki o hrvatskim planinama")
            del results # oslobadanje memorije (SPARQL upiti mogu vratiti velike kolicine podataka, eksplicitnim brisanjem brze oslobodimo memoriju)
            return self.triples
            
        except Exception as e:
            self.logger.error(f"Greska pri izvlacenju planina: {e}")
            return []

    ###############################################################################################################################
    
    ### "pravopisna" prilagodba ###
    # uklanjanje dijakritika, whistespaceova, underscore umjesto razmaka u viseclanim nazivima
    
    # svojstva (properties) - hrvatski prijevod svojstava (bez dijakritika)
    def translate_property(self, wikidata_property: str) -> str:
        # definiramo hrvatski prijevod svojstava
        translations = {
            "http://www.wikidata.org/prop/direct/P17": "drzava",
            "http://www.wikidata.org/prop/direct/P131": "nalazi_se_u", 
            "http://www.wikidata.org/prop/direct/P625": "koordinate",
            "http://www.wikidata.org/prop/direct/P1082": "broj_stanovnika",
            "http://www.wikidata.org/prop/direct/P571": "datum_osnivanja",
            "http://www.wikidata.org/prop/direct/P281": "postanski_broj",
            "http://www.wikidata.org/prop/direct/P37": "sluzbeni_jezik",
            "http://www.wikidata.org/prop/direct/P27": "drzavljanstvo",
            "http://www.wikidata.org/prop/direct/P19": "mjesto_rodjenja",
            "http://www.wikidata.org/prop/direct/P106": "zanimanje",
            "http://www.wikidata.org/prop/direct/P569": "datum_rodjenja",
            "http://www.wikidata.org/prop/direct/P20": "mjesto_smrti",
            "http://www.wikidata.org/prop/direct/P26": "bracni_partner",
            "http://www.wikidata.org/prop/direct/P69": "obrazovanje",
            "http://www.wikidata.org/prop/direct/P84": "arhitekt",
            "http://www.wikidata.org/prop/direct/P149": "arhitektonski_stil",
            "http://www.wikidata.org/prop/direct/P856": "sluzbena_stranica",
            "http://www.wikidata.org/prop/direct/P1448": "sluzbeni_naziv",
            "http://www.wikidata.org/prop/direct/P159": "sjediste",
            "http://www.wikidata.org/prop/direct/P1037": "ravnatelj",
            "http://www.wikidata.org/prop/direct/P101": "podrucje_rada",
            "http://www.wikidata.org/prop/direct/P859": "sponzor",
            "http://www.wikidata.org/prop/direct/P664": "organizator",
            "http://www.wikidata.org/prop/direct/P710": "sudionik",
            "http://www.wikidata.org/prop/direct/P527": "sastoji_se_od",
            "http://www.wikidata.org/prop/direct/P1174": "godisnji_broj_posjetitelja",
            "http://www.wikidata.org/prop/direct/P50": "pisac",
            "http://www.wikidata.org/prop/direct/P136": "zanr",
            "http://www.wikidata.org/prop/direct/P921": "tema",
            "http://www.wikidata.org/prop/direct/P577": "datum_izdavanja",
            "http://www.wikidata.org/prop/direct/P166": "nagrada",
            "http://www.wikidata.org/prop/direct/P123": "izdavac",
            "http://www.wikidata.org/prop/direct/P175": "izvodac",
            "http://www.wikidata.org/prop/direct/P1476": "naziv",
            "http://www.wikidata.org/prop/direct/P2047": "trajanje",
            "http://www.wikidata.org/prop/direct/P162": "producent",
            "http://www.wikidata.org/prop/direct/P86": "skladatelj",
            "http://www.wikidata.org/prop/direct/P1303": "svira_instrument",
            "http://www.wikidata.org/prop/direct/P2046": "povrsina",
            "http://www.wikidata.org/prop/direct/P610": "najvisa_tocka",
            "http://www.wikidata.org/prop/direct/P885": "izvor",
            "http://www.wikidata.org/prop/direct/P403": "usce",
            "http://www.wikidata.org/prop/direct/P4552": "gorje",
            "http://www.wikidata.org/prop/direct/P2044": "nadmorska_visina"
        }
        
        # ako postoji prijevod u rjecniku - dohvatit ce ga; ako ne - vratit ce ono sto je na kraju stringa poslije forwardslasha 
        # npr. vratio bi P17 da nema prijevoda "drzava"
        return translations.get(wikidata_property, wikidata_property.split('/')[-1])
    
    # uklanjanje dijakritika
    def remove_diacritics(self, text: str) -> str:
        replacements = {
            'č': 'c', 'Č': 'C',
            'ć': 'c', 'Ć': 'C', 
            'ž': 'z', 'Ž': 'Z',
            'š': 's', 'Š': 'S',
            'đ': 'd', 'Đ': 'D',
            'dž': 'dz', 'Dž': 'Dz', 'DŽ': 'DZ'
        }
        
        # dž/Dž (digraf)
        text = text.replace('dž', 'dz').replace('Dž', 'Dz').replace('DŽ', 'DZ')
        
        # ostali znakovi
        for original, replacement in replacements.items():
            text = text.replace(original, replacement)
            
        return text
    
    # zavrsno "ciscenje" naziva
    def clean_name(self, name: str) -> str:
        if name is None or not isinstance(name, str):
            return ""
        if name.startswith('http'):
            name = name.split('/')[-1]
        
        name = name.strip()                     # uklanjanje whitespaceove
        name = self.remove_diacritics(name)     # uklanjanje dijakritike
        name = name.replace(' ', '_')           # mijenjanje razmake s underscoreom (zbog prakticnosti)
        
        return name
    
    ###############################################################################################################################

    # izvlacenje podataka
    def extract_all_croatian_data(self):
        self.logger.info("Zapocinje izvlacenje svih hrvatskih podataka...")
        
        # ocisti postojece podatke
        self.triples = []
        self.entities = set()
        self.relations = set()
        
        # izvlacenje razlicitih tipova podataka s manjim limitima i pauzama
        self.extract_croatian_cities(limit=150)
        time.sleep(3)  # pauza izmedu poziva
        
        try:
            self.extract_croatian_people(limit=50)  # smanjen limit
            time.sleep(3)
        except Exception as e:
            self.logger.warning(f"Preskacemo osobe zbog greske: {str(e)}")
        
        try:
            self.extract_croatian_landmarks(limit=50)  # smanjen limit
        except Exception as e:
            self.logger.warning(f"Preskacemo znamenitosti zbog greske: {str(e)}")
        
        try:
            self.extract_croatian_universities(limit=50)  # smanjen limit
        except Exception as e:
            self.logger.warning(f"Preskacemo sveucilista zbog greske: {str(e)}")
        
        try:
            self.extract_croatian_institutes(limit=50)  # smanjen limit
        except Exception as e:
            self.logger.warning(f"Preskacemo institute zbog greske: {str(e)}")
        
        try:
            self.extract_croatian_festivals(limit=50)  # smanjen limit
        except Exception as e:
            self.logger.warning(f"Preskacemo festivale zbog greske: {str(e)}")
        
        try:
            self.extract_croatian_world_heritage_sites(limit=50)  # smanjen limit
        except Exception as e:
            self.logger.warning(f"Preskacemo UNESCO lokalitete svjetske bastine zbog greske: {str(e)}")

        try:
            self.extract_croatian_books(limit=50)  # smanjen limit
        except Exception as e:
            self.logger.warning(f"Preskacemo knjige zbog greske: {str(e)}")
        
        try:
            self.extract_croatian_music_and_performing_arts(limit=50)  # smanjen limit
        except Exception as e:
            self.logger.warning(f"Preskacemo glazbenu i scensku umjetnost zbog greske: {str(e)}")
        
        try:
            self.extract_croatian_national_parks(limit=50)  # smanjen limit
        except Exception as e:
            self.logger.warning(f"Preskacemo nacionalne parkove zbog greske: {str(e)}")
        
        try:
            self.extract_croatian_islands(limit=50)  # smanjen limit
        except Exception as e:
            self.logger.warning(f"Preskacemo otoke zbog greske: {str(e)}")
        
        try:
            self.extract_croatian_rivers(limit=50)  # smanjen limit
        except Exception as e:
            self.logger.warning(f"Preskacemo rijeke zbog greske: {str(e)}")
        
        try:
            self.extract_croatian_mountains(limit=50)  # smanjen limit
        except Exception as e:
            self.logger.warning(f"Preskacemo planine zbog greske: {str(e)}")
        
        self.logger.info(f"Ukupno izvuceno {len(self.triples)} trojki")
    
    # mapiranje entiteta/relacija i ID-jeva
    def create_mappings(self):
        self.entity2id = {entity: i for i, entity in enumerate(sorted(self.entities))}
        self.relation2id = {relation: i for i, relation in enumerate(sorted(self.relations))}
        self.id2entity = {i: entity for entity, i in self.entity2id.items()}
        self.id2relation = {i: relation for relation, i in self.relation2id.items()}
        
        self.logger.info(f"Kreirano {len(self.entities)} entiteta i {len(self.relations)} relacija")
    
    # spremanje podataka u datoteke
    def save_to_files(self, output_dir: str = "../../data/processed/"):
        try:
            output_dir = os.path.abspath(output_dir)
            os.makedirs(output_dir, exist_ok=True) # stvara direktorij ako ga nema
        
            # spremanje trojki
            triples_path = os.path.join(output_dir, "croatian_triples.txt")
            with open(triples_path, 'w', encoding='utf-8') as f:
                for s, p, o in self.triples:
                    f.write(f"{s}\t{p}\t{o}\n")
        
            # spremanje mapiranja
            entity_path = os.path.join(output_dir, "entity2id.json")
            with open(entity_path, 'w', encoding='utf-8') as f:
                json.dump(self.entity2id, f, ensure_ascii=False, indent=2)
            
            relation_path = os.path.join(output_dir, "relation2id.json")
            with open(relation_path, 'w', encoding='utf-8') as f:
                json.dump(self.relation2id, f, ensure_ascii=False, indent=2)
        
            # spremanje statistika (broj trojki, entiteta, relacija)
            stats = {
                "num_triples": len(self.triples),
                "num_entities": len(self.entities),
                "num_relations": len(self.relations),
                "sample_triples": self.triples[:10]
            }
        
            stats_path = os.path.join(output_dir, "stats.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)
        
            self.logger.info(f"Podaci spremljeni u {output_dir}")

        except Exception as e:
            self.logger.error(f"Greska pri spremanju datoteka: {str(e)}")
            raise
    
    # statistike o podatcima
    def get_statistics(self) -> Dict:
        return {
            "broj_trojki": len(self.triples),
            "broj_entiteta": len(self.entities),
            "broj_relacija": len(self.relations),
            "sample_entiteti": list(self.entities)[:10],
            "sample_relacije": list(self.relations)[:10],
            "sample_trojke": self.triples[:5]
        }

###############################################################################################################################

if __name__ == "__main__":
    # testiranje s Wikidata
    processor = CroatianKGProcessor()
    
    # izvlacenje hrvatskih podataka
    processor.extract_all_croatian_data()
    
    # kreiranje mapiranja
    processor.create_mappings()
    
    # spremanje podataka
    processor.save_to_files()
    
    # ispis statistika
    stats = processor.get_statistics()
    print("\n=== STATISTIKE HRVATSKOG GRAFA ZNANJA (WIKIDATA) ===")
    for key, value in stats.items():
        if isinstance(value, list) and len(value) > 3:
            print(f"{key}: {value[:3]}...")
        else:
            print(f"{key}: {value}")

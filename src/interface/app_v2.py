"""
    Gradio sucelje - inacica 2 prikazuje i odgovor direktnom pretragom grafa (bazna linija) i predikcije treniranog box-embedding modela, jedno pored drugog radi usporedbe.
"""

import os
import sys
import json
import torch
import gradio as gr
from datetime import datetime, date

# import iz src/data_processing i src/models bez pretvaranja u paket
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_processing"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models"))

from logical_query_generator import LogicalQueryGenerator
from box_embedding import BoxEmbedding
from train_q2b import answer_query_box

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRIPLES_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed", "croatian_triples.txt")
ENTITY2ID_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed", "entity2id.json")
RELATION2ID_PATH = os.path.join(BASE_DIR, "..", "..", "data", "processed", "relation2id.json")
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "results", "models", "box_embedding.pt")

generator = LogicalQueryGenerator(TRIPLES_PATH)

with open(ENTITY2ID_PATH, encoding="utf-8") as f:
    entity2id = json.load(f)
with open(RELATION2ID_PATH, encoding="utf-8") as f:
    relation2id = json.load(f)
id2entity = {i: e for e, i in entity2id.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BoxEmbedding(
    num_entities=len(entity2id),
    num_relations=len(relation2id),
    embedding_dim=64,
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

###############################################################################################################################
# odgovor direktnom pretragom grafa (bazna linija) - vidi prijasnju verziju
# za 1p i 2p: koristi se projekcija, e1 je sidreni entitet
# za 2i: koriste se "obrnuti" rubovi - trazi se entitet koji ima relaciju r1 prema vrijednosti e1 I relaciju r2 prema vrijednosti e2 (presjek dva uvjeta)

def graph_lookup_answer(qtype, e1, r1, e2, r2):
    if qtype == "1p":
        if not e1 or not r1:
            return None, "Unesite entitet i relaciju za 1p upit."
        answers = generator.objects.get((e1, r1), set())
        nl = f"Koji je entitet povezan s entitetom '{format_display_names(e1)}' relacijom '{format_display_names(r1)}'?"

    elif qtype == "2p":
        if not e1 or not r1 or not r2:
            return None, "Unesite pocetni entitet te prvu i drugu relaciju za 2p upit."
        intermediates = generator.objects.get((e1, r1), set())
        answers = generator.project(intermediates, r2)
        answers.discard(e1)
        nl = f"Krenuvsi od entiteta '{format_display_names(e1)}' preko relacije '{format_display_names(r1)}' i relacije '{format_display_names(r2)}', koji je entitet na kraju?"

    elif qtype == "2i":
        if not e1 or not r1 or not e2 or not r2:
            return None, "Unesite oba entiteta i obje relacije za 2i upit."
        s1 = generator.subjects.get((r1, e1), set())
        s2 = generator.subjects.get((r2, e2), set())
        answers = s1 & s2
        nl = f"Koji entitet zadovoljava '{format_display_names(r1)}' = '{format_display_names(e1)}' I '{format_display_names(r2)}' = '{format_display_names(e2)}'?"

    else:
        return None, "Tip upita nije odgovarajuć."

    return nl, answers


###############################################################################################################################
# predikcije treniranog modela - rangira SVE entitete po udaljenosti od kutije upita i vraca top-K najblizih

def model_predictions(qtype, e1, r1, e2, r2, top_k=5):
    # sastavlja query tuple u istom formatu kojeg koristi LogicalQueryGenerator
    try:
        if qtype == "1p":
            query = ("1p", (e1, (r1,)))
        elif qtype == "2p":
            query = ("2p", (e1, (r1, r2)))
        elif qtype == "2i":
            query = ("2i", ((e1, (r1,)), (e2, (r2,))))
        else:
            return "Tip upita nije odgovarajuć."

        center, offset = answer_query_box(model, query, entity2id, relation2id, device)
    except KeyError as e:
        return f"Entitet ili relacija ne postoji u modelu: {e}"

    all_ids = torch.arange(len(entity2id), device=device)
    with torch.no_grad():
        distances = model.box_distance(center, offset, all_ids).squeeze()

    sorted_indices = torch.argsort(distances)[:top_k]
    lines = []
    for idx in sorted_indices:
        entity_name = id2entity[idx.item()]
        dist = distances[idx].item()
        lines.append(f"- {format_display_names(entity_name)}  (udaljenost: {dist:.3f})")

    return "\n".join(lines)


###############################################################################################################################
# spaja oba nacina odgovaranja za Gradio callback

# spaja sve rijeci entiteta/relacija underscoreom kao sto je u clean_name() u croatian_kg_processor.py
def join_words_with_underscores(value):
    if value is None:
        value = ""
    words = value.split()
    return "_".join(words)

# za unos entiteta/relacija u lowercaseu - pretvara sve unose u mala slova i usporeduje s onim u entity2id i relation2id
# (inace je unos malim slovima vracao da nema pronadenih odgovora u grafu)
def compare_input_with_stored(input_value, stored_names):
    input_value = join_words_with_underscores(input_value)

    if input_value in stored_names: # ako se unos vec nalazi negdje spremljeno u _2id tako kako je uneseno -> ok
        return input_value

    for stored_name in stored_names: 
        if input_value.lower() == stored_name.lower(): 
            return stored_name
    return input_value
    

def answer_query(qtype, e1, r1, e2, r2):
    e1 = compare_input_with_stored(e1, entity2id) if e1 else ""
    r1 = compare_input_with_stored(r1, relation2id) if r1 else ""
    e2 = compare_input_with_stored(e2, entity2id) if e2 else ""  
    r2 = compare_input_with_stored(r2, relation2id) if r2 else ""

    nl, graph_answers = graph_lookup_answer(qtype, e1, r1, e2, r2)

    if nl is None:
        # graph_answers je zapravo poruka o gresci u ovom slucaju
        return graph_answers, "", ""

    if not graph_answers:
        graph_result = "Nema pronađenih odgovora u grafu."
    else:
        graph_result = "\n".join(f"{format_display_names(a)}" for a in sorted(graph_answers))

    model_result = model_predictions(qtype, e1, r1, e2, r2)

    return nl, graph_result, model_result

###############################################################################################################################
# DOHVACANJE PODATAKA ZA DROPDOWN SELECTION (ZA ENTITETE I RELACIJE) I FORMATIRANJE NJIHOVIH NAZIVA

# formatiranje stringova 
# - ako se radi o datumu: vratit ce dd. mm. yyyy. umjesto generickog datetime formata
# - ako je viseclani naziv razdvojen underscoreom (definirano s clean_name() u croatian_kg_processor.py), umjesto underscore ispisuje razmak
def format_display_names(value):
    if isinstance(value, str) and value:
            try:
                date = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
                return date.strftime("%d. %m. %Y.")
            except ValueError:
                return value.replace("_", " ")
    return value

# za dropdown selection:
# - entiteti i relacije mogu se odabrati iz dropdowna
# - entiteti se mogu upisati i rucno
# - formatiranje naziva: zamjena undersocrea s razmakom, sortiranje dropdown opcija
subject_choices = []
object_choices = []
relation_choices = [(r.replace("_", " "), r) for r in sorted(relation2id.keys())] # za dropdown selection relacija

with open(TRIPLES_PATH, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")

        if len(parts) == 3:
            subject_option = (format_display_names(parts[0]), parts[0]) # (display name, stored name)
            object_option = (format_display_names(parts[2]), parts[2])

            if subject_option not in subject_choices:
                subject_choices.append(subject_option)
            if object_option not in object_choices:
                object_choices.append(object_option)

subject_choices.sort()
object_choices.sort()

# definiranje atributa dropdowna ovisno o tipu upita
# ako se odabere 2i upit, prvi input se bira medu objektima; inace (1p, 2p) medu subjektima
def update_query_fields(query_type):
    if query_type == "1p":
        choices = subject_choices
        entity_label = "Entitet"
        relation_label = "Svojstvo"
        second_relation_label = "Druga relacija"

    elif query_type == "2p":
        choices = subject_choices
        entity_label = "Početni entitet"
        relation_label = "Prva relacija" 
        second_relation_label = "Druga relacija"
        
    else:
        choices = object_choices
        entity_label = "Vrijednost prvog uvjeta"
        relation_label = "Svojstvo prvog uvjeta"
        second_relation_label = "Svojstvo drugog uvjeta"

    first_entity = gr.Dropdown(
        choices=choices,
        label=entity_label,
        value=None,
        allow_custom_value=True,
        elem_classes="input-field"
    )
    first_relation = gr.Dropdown(
        label=relation_label,
        choices=relation_choices,
        value=None,
        elem_classes="input-field"
    )
    second_entity = gr.Dropdown(
        label="Vrijednost drugog uvjeta",
        choices=object_choices,
        value=None,
        allow_custom_value=True,
        elem_classes="input-field",
        visible=query_type == "2i" # samo za 2i upite ce bit vidljiv
    )
    second_relation = gr.Dropdown(
        label=second_relation_label,
        choices=relation_choices,
        value=None,
        elem_classes="input-field",
        visible=query_type != "1p" # vidljivo samo za 2p i 2i upite
    )

     # ovo je vidljivo za 2p i 2i upite, ali je napravljeno iskljucivo da bi za odabrani 2p upit, gr.Row s relacijama bili jedno ispod drugog
     # (za 2p ce bit nevidljivo)
    row_update = gr.Row(visible=query_type != "1p")

    return first_entity, first_relation, second_entity, second_relation, row_update

###############################################################################################################################
# SUCELJE

theme = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="emerald",
    neutral_hue="stone",
)

CSS_PATH = os.path.join(BASE_DIR, "style.css")
with open(CSS_PATH, encoding="utf-8") as f:
    css = f.read()

with gr.Blocks(title="Croatian Query2Box") as demo:
    # ako browser u kojemu se otvori app ima postavljen dark theme, boje postavljene u app-u i dark theme ce se clashati
    # ovo postavlja light temu za app
    demo.load(
        js="""
        function forceLight() {
            const url = new URL(window.location);
            if (url.searchParams.get('__theme') !== 'light') {
                url.searchParams.set('__theme', 'light');
                window.location.href = url.href;
            }
        }
        """
    )

    gr.Markdown(
        "# Croatian Query2Box\n"
        "Usporedba odgovora dobivenog direktnom pretragom grafa (bazna linija) i predikcija treniranog box-embedding modela.",
        elem_id="main-title"
    )
    gr.Markdown("---", elem_id="title-separator") # ravna linija ispod naslova da dijelovi app budu vizualno odijeljeni

    qtype = gr.Radio(["1p", "2p", "2i"], value="1p", label="Tip logičkog upita")

    # inicijalni izgled input polja (kad se app tek otvori)
    with gr.Row(elem_classes="input-row"):
        with gr.Column(scale=1, min_width=160):
            e1 = gr.Dropdown(
                label="Entitet", 
                choices=subject_choices, 
                value=None,              
                allow_custom_value=True,
                elem_classes="input-field") 
        with gr.Column(scale=1, min_width=160):
            r1 = gr.Dropdown(
            label="Svojstvo", 
            choices=relation_choices, 
            value=None, 
            elem_classes="input-field")

    with gr.Row(elem_classes="input-row", visible=False) as second_row:
        with gr.Column(scale=1, min_width=160):
            e2 = gr.Dropdown(
                label="Vrijednost drugog uvjeta", 
                choices=object_choices, 
                value=None,              
                allow_custom_value=True,
                elem_classes="input-field",
                visible=False)
        with gr.Column(scale=1, min_width=160):
            r2 = gr.Dropdown(
                label="Druga relacija", 
                choices=relation_choices, 
                value=None, 
                elem_classes="input-field",
                visible=False)

    with gr.Row(elem_id="button-row"):
        clear_btn = gr.ClearButton(
            [e1, e2, r1, r2],
            value="Poništi unose",
            elem_classes="my-button"
        )
        submit_btn = gr.Button("Pronađi odgovor", variant="primary", elem_classes="my-button")

    #gr.Markdown("&nbsp;") # prazan red da dijelovi app budu vizualno odijeljeni
    gr.Markdown("---")
    nl_output = gr.Textbox(label="Upit (prirodni jezik)", interactive=False)

    with gr.Row():
        graph_output = gr.Textbox(label="Odgovor iz grafa (točno)", interactive=False, lines=6)
        model_output = gr.Textbox(label="Top-5 predikcija modela", interactive=False, lines=6)

    submit_btn.click(
        fn=answer_query,
        inputs=[qtype, e1, r1, e2, r2],
        outputs=[nl_output, graph_output, model_output],
    )

    # ažuriranje dropdown selectiona kad se promijeni tip upita
    qtype.change(
        fn=update_query_fields,
        inputs=[qtype],
        outputs=[e1, r1, e2, r2, second_row]
    )

    gr.Markdown(
        "Napomena:  " \
        "\nPrimjer 1p upita: Entitet 1 = 'Zagreb', Relacija 1 = 'postanski_broj'.  "
        "\nPrimjer 2p upita: Entitet 1 = 'Stjepan_Mesic', Relacija 1 = 'mjesto rodjenja', Relacija 2 = 'postanski broj'.  "
        "\nPrimjer 2i upita: (Entitet 1 = 'pisac' = Relacija 1 = 'zanimanje') I (Entitet 2 = 'novinar' = Relacija 2 = 'zanimanje')."
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=False,
        css=css,
        theme=theme,
    )
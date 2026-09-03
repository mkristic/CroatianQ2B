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

# formatiranje stringova za ispis
# - ako se radi o datumu: vratit ce dd. mm. yyyy. umjesto generickog datetime formata
# - ako je viseclani naziv razdvojen underscoreom (definirano s clean_name() u croatian_kg_processor.py), umjesto underscore ispisuje razmak
def to_display(value):
    if isinstance(value, str) and value:
            try:
                date = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
                return date.strftime("%d. %m. %Y.")
            except ValueError:
                return value.replace("_", " ")
    return value

###############################################################################################################################
# odgovor direktnom pretragom grafa (bazna linija) - vidi prijasnju verziju
# za 1p i 2p: koristi se projekcija, e1 je sidreni entitet
# za 2i: koriste se "obrnuti" rubovi - trazi se entitet koji ima relaciju r1 prema vrijednosti e1 I relaciju r2 prema vrijednosti e2 (presjek dva uvjeta)

def graph_lookup_answer(qtype, e1, r1, e2, r2):
    if qtype == "1p":
        if not e1 or not r1:
            return None, "Unesite entitet i relaciju za 1p upit."
        answers = generator.objects.get((e1, r1), set())
        nl = f"Koji je entitet povezan s entitetom '{to_display(e1)}' relacijom '{to_display(r1)}'?"

    elif qtype == "2p":
        if not e1 or not r1 or not r2:
            return None, "Unesite pocetni entitet te prvu i drugu relaciju za 2p upit."
        intermediates = generator.objects.get((e1, r1), set())
        answers = generator.project(intermediates, r2)
        answers.discard(e1)
        nl = f"Krenuvsi od entiteta '{to_display(e1)}' preko relacije '{to_display(r1)}' i relacije '{to_display(r2)}', koji je entitet na kraju?"

    elif qtype == "2i":
        if not e1 or not r1 or not e2 or not r2:
            return None, "Unesite oba entiteta i obje relacije za 2i upit."
        s1 = generator.subjects.get((r1, e1), set())
        s2 = generator.subjects.get((r2, e2), set())
        answers = s1 & s2
        nl = f"Koji entitet zadovoljava '{to_display(r1)}' = '{to_display(e1)}' I '{to_display(r2)}' = '{to_display(e2)}'?"

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
        lines.append(f"- {to_display(entity_name)}  (udaljenost: {dist:.3f})")

    return "\n".join(lines)


###############################################################################################################################
# spaja oba nacina odgovaranja za Gradio callback

def answer_query(qtype, e1, r1, e2, r2):
    e1 = e1.strip() if e1 else ""
    r1 = r1.strip() if r1 else ""
    e2 = e2.strip() if e2 else ""
    r2 = r2.strip() if r2 else ""

    nl, graph_answers = graph_lookup_answer(qtype, e1, r1, e2, r2)

    if nl is None:
        # graph_answers je zapravo poruka o gresci u ovom slucaju
        return graph_answers, "", ""

    if not graph_answers:
        graph_result = "Nema pronađenih odgovora u grafu."
    else:
        graph_result = "\n".join(f"{to_display(a)}" for a in sorted(graph_answers))

    model_result = model_predictions(qtype, e1, r1, e2, r2)

    return nl, graph_result, model_result


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

with gr.Blocks(title="Croatian Query2Box", css=css, theme=theme) as demo:
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

    relation_choices = [(r.replace("_", " "), r) for r in sorted(relation2id.keys())] # za dropdown selection

    with gr.Row(elem_classes="input-row"):
        e1 = gr.Textbox(label="Entitet 1 (sidrena vrijednost za 1p/2p)", elem_classes="input-field")
        r1 = gr.Dropdown(label="Relacija 1", choices=relation_choices, value=None, elem_classes="input-field")

    with gr.Row(elem_classes="input-row"):
        e2 = gr.Textbox(label="Entitet 2 (samo za 2i)", elem_classes="input-field")
        r2 = gr.Dropdown(label="Relacija 2 (za 2p i 2i)", choices=relation_choices, value=None, elem_classes="input-field")

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

    gr.Markdown(
        "Napomena:  " \
        "\nNazivi entiteta/relacija moraju odgovarati onima u croatian_triples.txt (bez dijakritika, razmak -> underscore).  " \
        "\nPrimjer 1p upita: Entitet 1 = 'Zagreb', Relacija 1 = 'postanski_broj'.  "
        "\nPrimjer 2p upita: Entitet 1 = 'Stjepan_Mesic', Relacija 1 = 'mjesto rodjenja', Relacija 2 = 'postanski broj'.  "
        "\nPrimjer 2i upita: (Entitet 1 = 'pisac' = Relacija 1 = 'zanimanje') I (Entitet 2 = 'novinar' = Relacija 2 = 'zanimanje')."
    )

if __name__ == "__main__":
    demo.launch()
"""
    Gradio sucelje
"""

import os
import sys
import gradio as gr

# import is src/data_processing bez pretvaranja u paket
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_processing"))
from logical_query_generator import LogicalQueryGenerator 

TRIPLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "processed", "croatian_triples.txt")

generator = LogicalQueryGenerator(TRIPLES_PATH)

###############################################################################################################################
# logika odgovaranja na upite

# racuna odgovor na upit ovisno o odabranom tipu (1p/2p/2i)
# za 1p i 2p: koristi se projekcija, e1 je sidreni entitet
# za 2i: koriste se "obrnuti" rubovi, tj. trazi se entitet koji ima relaciju r1 prema e1 I relaciju r2 prema e2
# return vraca tuple od 2 stringa (nl_output, answer_output)
def answer_query(qtype, e1, r1, e2, r2):
    e1 = e1.strip()
    e2 = e2.strip()
    r1 = r1.strip()
    r2 = r2.strip()
    
    if qtype == "1p":
        if not e1 or not r1:
            return "Unesite entitet i relaciju za 1p upit.", ""
        answers = generator.objects.get((e1, r1), set())
        nl = f"Koji je entitet povezan s entitetom '{e1}' relacijom '{r1}'?"
        
    elif qtype == "2p":
        if not e1 or not r1 or not r2:
            return "Unesite pocetni entitet te prvu i drugu relaciju za 2p upit.", ""
        intermediates = generator.objects.get((e1, r1), set())
        answers = generator.project(intermediates, r2)
        answers.discard(e1)
        nl = f"Krenuvsi od entiteta '{e1}' preko relacije '{r1}' i relacije '{r2}', koji je entitet na kraju?"
    
    elif qtype == "2i":
        if not e1 or not r1 or not e2 or not r2:
            return "Unesite oba entiteta i obje relacije za 2i upit.", ""
        s1 = generator.subjects.get((r1, e1), set())
        s2 = generator.subjects.get((r2, e2), set())
        answers = s1 & s2
        nl = f"Koji entitet zadovoljava '{r1}' = '{e1}' I '{r2}' = '{e2}'?"
    
    else:
        return "Tip upita nije odgovarajuc.", ""
    
    if not answers:
        result = "Nema pronadenih odgovora u grafu. Provjerite nazive (rabe se hrvatski nazivi bez dijakritika)."
    else:
        result = "\n".join(f"{ans}" for ans in sorted(answers))
        
    return nl, result
    
###############################################################################################################################
# sucelje

with gr.Blocks(title="Croatian Query2Box") as demo:
    qtype = gr.Radio(["1p", "2p", "2i"], value="1p", label="Tip logickog upita")
    
    with gr.Row():
        e1 = gr.Textbox(label="Entitet 1 (sidro za 1p/2p, vrijednost svojstva za 2i)")
        r1 = gr.Textbox(label="Relacija 1")
    
    with gr.Row():
        e2 = gr.Textbox(label="Entitet 2 (samo za 2p)")
        r2 = gr.Textbox(label="Relacija 2 (za 2p i 2i)")
    
    submit_btn = gr.Button("Odgovor", variant="primary")
    
    nl_output = gr.Textbox(label="Upit (prirodni jezik)", interactive=False)
    answer_output = gr.Textbox(label="Odgovor", interactive=False, lines=6)
    
    submit_btn.click(
        fn=answer_query,
        inputs=[qtype, e1, r1, e2, r2],
        outputs=[nl_output, answer_output]
    )
    
    gr.Markdown(
        "Napomena: nazivi entiteta/relacija moraju odgovarati onima u croatian_triples.txt (bez dijakritika, razmak -> underscore"
        "\nPrimjer 1p upita: Entitet 1 = 'Mali_Losinj', Relacija 1 = 'broj_stanovnika'"
    )
    
if __name__ == "__main__":
    demo.launch()
    
        
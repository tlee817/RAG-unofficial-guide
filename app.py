"""
app.py — Gradio web interface for the UCLA CS Unofficial Guide RAG system.

Run: python app.py
Then open: http://localhost:7860
"""

import gradio as gr
from query import ask

EXAMPLES = [
    "What are students' biggest complaints about Eggert's courses?",
    "Is Reinman a good professor for CS33?",
    "What is the best way to prepare for Eggert's exams?",
    "How difficult is CS111 compared to other CS courses?",
    "What do students say about Nacherberg's CS131?",
]


def handle_query(question: str):
    if not question.strip():
        return "Please enter a question.", ""
    result = ask(question)
    return result["answer"], result["sources"]


with gr.Blocks(title="UCLA CS Unofficial Guide") as demo:
    gr.Markdown("## UCLA CS Unofficial Guide")
    gr.Markdown(
        "Ask anything about UCLA CS professors and courses. "
        "Answers are grounded in real Bruinwalk student reviews."
    )

    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="Your question",
                placeholder="e.g. Is Eggert's CS33 worth taking?",
                lines=2,
            )
            ask_btn = gr.Button("Ask", variant="primary")

    answer = gr.Textbox(label="Answer", lines=8, interactive=False)
    sources = gr.Textbox(label="Retrieved from", lines=4, interactive=False)

    gr.Examples(examples=EXAMPLES, inputs=question)

    ask_btn.click(handle_query, inputs=question, outputs=[answer, sources])
    question.submit(handle_query, inputs=question, outputs=[answer, sources])

if __name__ == "__main__":
    demo.launch()

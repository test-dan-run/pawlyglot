import gradio as gr
from orchestrator_stream import run_pipeline

with gr.Blocks() as demo:
    gr.Markdown("Pawlyglot")
    with gr.Row():
        with gr.Column():
            inp = gr.Audio(type="filepath")
            btn = gr.Button("Run")
        out = gr.Audio(streaming=True, autoplay=True)

    btn.click(fn=run_pipeline, inputs=inp, outputs=out)

demo.queue().launch()
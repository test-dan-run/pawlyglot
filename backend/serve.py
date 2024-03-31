import gradio as gr
from orchestrator_stream import run_pipeline

with gr.Blocks() as demo:
    gr.HTML(value="<h1 style='color: orange'>Pawlyglot</h1>")
    gr.Markdown(
    """
    **pawlyglot** is a project that aims to develop an end-to-end multilingual TTS, voice-cloning and lip-syncing pipeline, by combining several open-source projects.<br>
    Visit the development git repository to see the behind-the-scenes [here](https://github.com/test-dan-run/pawlyglot).

    Notes:<br>
    Only **EN > CN** translations will be supported at the moment.<br>
    Current pipeline only contains: **VAD -> ASR -> MT -> VC/TTS** (Lip-syncing will be available in future versions)<br>
    You will only need to wait for the first VAD segment of audio to be synthesized before a response returns. The rest of the segments will come streaming in after. <br>

    Current Average RTF: 0.6
    """
                )
    with gr.Row():
        with gr.Column():
            inp = gr.Audio(type="filepath")
            btn = gr.Button("Run")
        out = gr.Audio(streaming=True, autoplay=True)

    btn.click(fn=run_pipeline, inputs=inp, outputs=out)

demo.queue().launch()
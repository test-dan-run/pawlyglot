import gradio as gr
from orchestrator_makeshift import run_complete_pipeline, extract_from_youtube_url

with gr.Blocks() as demo:
    
    gr.HTML(value="<h1 style='color: orange'>Pawlyglot (Beta)</h1>")
    gr.Markdown(
    """
    **pawlyglot** is a project that aims to develop an end-to-end multilingual TTS, voice-cloning and lip-syncing pipeline, by combining several open-source projects.<br>
    Visit the development git repository to see the behind-the-scenes [here](https://github.com/test-dan-run/pawlyglot).

    **Notes:**<br>
    Only **EN > CN** translations will be supported at the moment.<br>
    Current pipeline only contains: **VAD -> ASR -> MT -> VC/TTS** (Lip-syncing will be available in future versions)<br>
    You will only need to wait for the first VAD segment of audio to be synthesized before a response returns. The rest of the segments will come streaming in after. <br>
    
    **CAVEAT:**<br>
    Pipeline will assume there's only 1 speaker throughout the entire clip. <br>
    If there's 2 speakers or more, the pipeline combines the characteristics of the multiple speakers to generate a brand new voice.
    
    **Steps**<br>
    Step 1<br>
    (Option 1) Copy a Youtube Video URL and click download. This will populate the audio input below.<br>
    (Option 2) Put in an English audio clip from your own file directory.<br>
    Step 2<br>
    Wait! Average Wait time: 6~10s for first output chunk<br>
    There might be multiple people trying the pipeline out. Using Gradio's default queueing system to queue requests. Be patient ya!
    """
                )
    with gr.Row():
        with gr.Column():
            with gr.Row():
                inp_youtube_url = gr.Textbox(label="Youtube URL", interactive=True, scale=3)
                youtube_btn = gr.Button("Download", scale=1)
            inp_video = gr.Video()
            btn = gr.Button("Run")
            # gr.Examples("/examples", inp_audio)
        with gr.Column():
            out_video = gr.Video(label="Lip Sync")

    youtube_btn.click(fn=extract_from_youtube_url, inputs=inp_youtube_url, outputs=inp_video)
    btn.click(fn=run_complete_pipeline, inputs=inp_video, outputs=out_video)

demo.queue().launch(server_name="0.0.0.0", server_port=7860,
                    # share=True, auth=("nlp", "cuteboi"),
                    )

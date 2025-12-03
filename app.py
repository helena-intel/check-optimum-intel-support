import importlib.metadata
import gradio as gr
from optimum_support import show_is_supported

with gr.Blocks() as app:
    gr.Markdown("# Check if model is supported by optimum-intel[openvino]")
    with gr.Column():
        model_id = gr.Textbox(label="model_id")
        process_button = gr.Button("Check")
        output_text = gr.Markdown(label="result", height=100)
        optimum_intel_version = importlib.metadata.version("optimum-intel")
        gr.Markdown(
            f"Tested with optimum-intel {optimum_intel_version}. For testing purposes only, results may be wrong."
        )

    process_button.click(show_is_supported, inputs=[model_id], outputs=output_text)
    model_id.submit(show_is_supported, inputs=[model_id], outputs=output_text)

app.launch()

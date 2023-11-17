import gradio as gr
from utils import watsonx

def call_watsonxai (
    user_input
):
    return watsonx.get_watsonxai_response(input=user_input)

with gr.Blocks() as demo:
    with gr.Row():
        user_input = gr.Textbox(label="Input Placeholder", placeholder="Insert text here...")

    with gr.Row():
        action_btn = gr.Button("Submit", variant='primary')

    with gr.Row():
        output = gr.Textbox(label="Output Text")

    action_btn.click(
      fn=call_watsonxai,
      inputs=[user_input],
      outputs=output,
      api_name="submit")

demo.launch(share=True)

import gradio as gr
import os
import json
from utils import watsonx

with open('app/example.txt', 'r') as file:
    example_text = file.read()
examples = {'Forest': example_text}
def example_lookup(text):
    if text:
        return examples[text]
    return ''

example_instruction = "Please provide a summary of the following text. Do not add any information that is not mentioned in the text below."

def clear_out():
    cleared_tuple = (gr.Textbox.update(value=""),
                    gr.Textbox.update(value=""),
                    gr.Textbox.update(value=""),
                    gr.Textbox.update(value=""))
    return cleared_tuple

# list of LLM models to use for text summarization
models = ['ibm/granite-13b-chat-v1', 'meta-llama/llama-2-70b-chat']

# call watsonx.py > get_watsonxai_response method
def summarize(modelId, input_text, custom_instruction, max_new_tokens, temperature, top_p):
    result = watsonx.get_watsonxai_response(
        modelId, input_text, custom_instruction, max_new_tokens, temperature, top_p
    )
    return result.strip('<|endoftext|>')

# app design
with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown("# IBM watsonx.ai - Text Summarization")
        example_holder = gr.Textbox(visible=False, label="Input Text", value="example")
    with gr.Row():
        modelId = gr.Dropdown(label="Choose a Model:", choices=models, value='ibm/granite-13b-chat-v1')
    with gr.Row():
        # input column
        with gr.Column(scale=5):
            custom_instruction = gr.Textbox(label="Input your prompt instruction", value=example_instruction)
            input_text = gr.Textbox(label="Input your text", placeholder="Insert some long text here...")
            example = gr.Examples(examples=[[example_instruction, "Forest"]], inputs=[custom_instruction, example_holder])
        # options and actions column
        with gr.Column(scale=3):
            with gr.Accordion("Advanced Generation Options", open=False):
                max_new_tokens = gr.Slider(minimum=0, maximum=4096, step=1, value=512, label="Max Tokens")
                temperature = gr.Slider(minimum=0.01, maximum=2.0, step=0.01, value=0.7, label="Temperature")
                top_p = gr.Slider(minimum=0, maximum=1.0, step=0.01, value=1.0, label="Top P")
            summarize_btn = gr.Button("Summarize", variant='primary')
            reset_btn = gr.Button("Reset")
        # output column
        with gr.Column(scale=4):
          output = gr.Textbox(label="Output Text")

    summarize_btn.click(
        fn=summarize,
        inputs=[modelId, input_text, custom_instruction, max_new_tokens, temperature, top_p],
        outputs=output,
        api_name="summarize")

    reset_btn.click(
        fn=clear_out,
        inputs=[],
        outputs=[input_text, output, example_holder, custom_instruction],
        show_progress=False)

    example_holder.change(
        fn=example_lookup,
        inputs=example_holder,
        outputs=input_text,
        show_progress=False)
# end app design

# launch app
'''
demo.launch(server_port=int(os.getenv('CDSW_APP_PORT')),
           enable_queue=True,
           show_error=True,
           server_name='127.0.0.1',
)
'''
demo.launch(share=True)
# end app

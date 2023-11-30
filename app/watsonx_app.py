import gradio as gr
import os
from utils import watsonx

with open('app/example.txt', 'r') as file:
    example_text = file.read()

examples = {'Example Text': example_text}

def example_lookup(text):
  if text:
    return examples[text]
  return ''

example_instruction = "Please provide a summary of the following text. Do not add any information that is not mentioned in the text below."

def clear_out():
  cleared_tuple = (gr.Textbox.update(value=""), gr.Textbox.update(value=""), gr.Textbox.update(value=""))
  return cleared_tuple

def summarize(input_text, instruction_text, max_new_tokens):
    return watsonx.get_watsonxai_response(
        input_text=input_text,
        instruction=instruction_text,
        max_new_tokens=max_new_tokens
    )

with gr.Blocks() as demo:
  with gr.Row():
    gr.Markdown("# IBM watsonx.ai - Text Summarization using Granite model")
    example_holder = gr.Textbox(visible=False, label="Input Text", value="example")
  with gr.Row():
    with gr.Column(scale=6):
      custom_instruction = gr.Textbox(label="Input your prompt instruction:", value=example_instruction)
      input_text = gr.Textbox(label="Input your text", placeholder="Insert some long text here...")
      example = gr.Examples(examples=[[example_instruction, "Example Text"]], inputs=[custom_instruction, example_holder])
    with gr.Column(scale=3):
      with gr.Accordion("Advanced Generation Options", open=False):
        max_new_tokens = gr.Slider(minimum=0, maximum=4096, step=1, value=512, label="Max Tokens")
      summarize_btn = gr.Button("Summarize", variant='primary')
      reset_btn = gr.Button("Reset")
    with gr.Column(scale=3):
      output = gr.Textbox(label="Output Text")

  summarize_btn.click(
    fn=summarize,
    inputs=[input_text, custom_instruction, max_new_tokens],
    outputs=output,
    api_name="summarize")

  reset_btn.click(
    fn=clear_out,
    inputs=[],
    outputs=[input_text, output, example_holder, custom_instruction],
    show_progress=False)

  example_holder.change(fn=example_lookup, inputs=example_holder, outputs=input_text, show_progress=False)

demo.launch(share=True)

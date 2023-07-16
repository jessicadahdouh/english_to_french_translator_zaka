import gradio as gr


def translate(text):

    return "Hello " + text + "!"


input_text = gr.Textbox(label="English text", placeholder="Enter English text here")
output_text = gr.Textbox(label="French text")

demo = gr.Interface(fn=translate,
                    inputs=input_text,
                    outputs=output_text,
                    title="Translator",
                    examples=[["Hello World!"], ["He is playing outside."], ["She is sad."]])

demo.launch()

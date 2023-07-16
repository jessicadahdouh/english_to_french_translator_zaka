from flask import Flask, render_template
from EnToFrProject.settings import ProjectConfig, Config
from model.implement_translation import translate_text
import gradio as gr
import asyncio


templates_folder = ProjectConfig.templates_folder
static_folder = ProjectConfig.static_folder

app = Flask(__name__, template_folder=templates_folder, static_folder=static_folder)

app.config.from_object(Config)


def translate(input_text):
    translate_text(input_text, model="Bidirectional")
    return input_text


async def run_gradio():
    input_text = gr.Textbox(label="English text", placeholder="Enter English text here")
    output_text = gr.Textbox(label="French text")

    iface = gr.Interface(fn=translate,
                         inputs=input_text,
                         outputs=output_text,
                         title="Translator",
                         allow_flagging=False,
                         examples=[["Hello World!"], ["He is playing outside."], ["She is sad."]])

    await iface.launch(share=True)


@app.route("/")
def gradio_interface():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_gradio())

    return render_template("translator.html")


if __name__ == '__main__':
    app.run()

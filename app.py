from flask import Flask, render_template
from EnToFrProject.settings import ProjectConfig, Config
# from EnToFrProject.routes import translator_routes as translator_bp
import gradio as gr


templates_folder = ProjectConfig.templates_folder
static_folder = ProjectConfig.static_folder

app = Flask(__name__, template_folder=templates_folder, static_folder=static_folder)

# configure the project
app.config.from_object(Config)
# app.register_blueprint(translator_bp)

app = Flask(__name__)


def translate(input_text):
    return input_text


@app.route("/")
def gradio():
    input_text = gr.Textbox(label="English text", placeholder="Enter English text here")
    output_text = gr.Textbox(label="French text")

    demo = gr.Interface(fn=translate,
                        inputs=input_text,
                        outputs=output_text,
                        title="Translator",
                        examples=[["Hello World!"], ["He is playing outside."], ["She is sad."]])

    demo.launch()


if __name__ == '__main__':
    app.run()

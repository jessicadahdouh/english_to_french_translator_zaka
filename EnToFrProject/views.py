# from model.implement_translation import translate_text
from flask import render_template
import gradio as gr


def gradio_interface():
    gr.Interface(
        fn=translate_text,
        inputs="text",
        outputs="text",
        examples=[
            ["I went to the supermarket yesterday."],
            ["Helen is a good swimmer."]
        ]
    ).launch(share=True)

def render_index():
    return render_template('translator.html')


def render_translator(text):
    translated_text = ""
    if text:
        # translated_text = translate_text(text=text)
        print("entered")
    response = {"text": translated_text}

    return render_template('translator.html', response=response)


def translate_text(input_text):
    # Perform translation logic here using the input_text
    translated_text = "Translated: " + input_text
    return translated_text


from flask import Blueprint, render_template, jsonify
from .views import translate_text, gradio_interface

translator_routes = Blueprint('translator_bp', __name__)


@translator_routes.route('/gradio-app')
def index():
    gradio_interface()
    return ''
    # return render_template('translator.html')

# @translator_routes.route('/translate', methods=['POST'])
# def translate():
#     data = request.get_json()
#     input_text = data.get('text')
#     translated_text = translate_text(input_text)
#     return jsonify({'translation': translated_text})

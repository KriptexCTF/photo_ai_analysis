from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename

from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

app = Flask(__name__)

# Настройки папок для загруженных и обработанных файлов
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'processed/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    # Рендеринг главной страницы
    return render_template('index.html')

@app.route('/upload')
def chat():
    # Рендеринг страницы чата
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Обработка загрузки файла
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    if file:
        global filename
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        return jsonify({'message': f'File processed as {filename}'})
    return jsonify({'message': 'File upload failed'}), 500

@app.route('/ask1', methods=['POST'])
def ask1():
    if ('filename' in globals()):
        question = f"YOU: {request.form.get('question')}"
        if question:
            return jsonify({'message': question})
        return jsonify({'message': 'No question provided'}), 400
    else:
        return jsonify({'message': 'Please upload picture!!!'}), 400
@app.route('/ask2', methods=['POST'])
def ask2():
    if ('filename' in globals()):
        raw_image = (Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))).convert('RGB')
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    
        # Обработка вопроса от пользователя
        question = request.form.get('question')
        if question:
            # Пример простого ответа на вопрос
            #answer = f"Your question was: {question} {filename}"
    
            inputs = processor(raw_image, question, return_tensors="pt")
            out = model.generate(**inputs, max_length=256, num_beams=5)
            answer = f"MIREA_GPT: {processor.decode(out[0], skip_special_tokens=True)}"
            #return jsonify({'question': question, 'message': answer})
            return jsonify({'message': answer})
    else:        
        return jsonify({'message': 'No question provided'}), 400

if __name__ == '__main__':
    app.run(debug=True)

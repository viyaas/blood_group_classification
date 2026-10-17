import os
import tempfile

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=None)
CORS(app)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
IMAGE_SIZE = (128, 128)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# MODEL_VARIANT selects which trained model to serve: "positive" (4 classes) or "all" (8 classes, less accurate).
MODELS = {
    'all': {
        'path': os.path.join(BASE_DIR, 'models', 'fingerprint_bloodgroup.h5'),
        'labels': ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-'],
    },
    'positive': {
        'path': os.path.join(BASE_DIR, 'models', 'fingerprint_bloodgroup_positive.h5'),
        'labels': ['A+', 'AB+', 'B+', 'O+'],
    },
}

MODEL_VARIANT = os.environ.get('MODEL_VARIANT', 'positive')
if MODEL_VARIANT not in MODELS:
    raise SystemExit(f"MODEL_VARIANT must be one of {sorted(MODELS)}, got '{MODEL_VARIANT}'")

MODEL_PATH = MODELS[MODEL_VARIANT]['path']
LABELS = MODELS[MODEL_VARIANT]['labels']

model = tf.keras.models.load_model(MODEL_PATH)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_blood_group(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
    prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)
    label = int(np.argmax(prediction, axis=1)[0])
    return {
        'label': label,
        'blood_group': LABELS[label] if label < len(LABELS) else 'Unknown',
        'confidence': float(prediction[0][label] * 100),
        'success': True,
    }


@app.route('/')
def index():
    return send_from_directory(os.path.join(BASE_DIR, 'frontend'), 'index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part', 'success': False}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file', 'success': False}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed', 'success': False}), 400

    # Uploaded images are processed in a temp file and never kept.
    suffix = os.path.splitext(secure_filename(file.filename))[1]
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        file.save(tmp.name)
        try:
            return jsonify(predict_blood_group(tmp.name))
        except Exception:
            app.logger.exception('Prediction failed')
            return jsonify({'error': 'Could not process image', 'success': False}), 400


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK', 'model_variant': MODEL_VARIANT})


if __name__ == '__main__':
    app.run(
        debug=os.environ.get('FLASK_DEBUG') == '1',
        host=os.environ.get('HOST', '127.0.0.1'),
        port=int(os.environ.get('PORT', 5000)),
    )

import glob
import sys

import tensorflow as tf

LABELS = {
    4: ['A+', 'AB+', 'B+', 'O+'],
    8: ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-'],
}

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'models/fingerprint_bloodgroup_positive.h5'
model = tf.keras.models.load_model(MODEL)
labels = LABELS[model.output_shape[-1]]

paths = sorted(p for p in glob.glob('uploads/*') if p.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))

for p in paths:
    x = tf.keras.utils.img_to_array(tf.keras.utils.load_img(p, target_size=(128, 128))) / 255.0
    probs = model.predict(x[None], verbose=0)[0]
    top = int(probs.argmax())
    print(f'{p.split("/")[-1]:60s} {labels[top]:4s} {probs[top] * 100:5.1f}%')

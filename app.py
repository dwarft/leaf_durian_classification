from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Urutan HARUS sesuai abjad nama folder dataset saat training
CLASS_NAMES = [
    'Antraknosa', 
    'Bercak_Algae', 
    'Hawar_Daun', 
    'Kanker_Batang', 
    'Non_Durian',  # Kelas Baru
    'Penyakit_Lain', 
    'Sehat'
]

DISEASE_INFO = {
    'Antraknosa': {
        'nama_lain': 'Bercak Daun Cokelat / Karat Daun',
        'penjelasan': 'Penyakit akibat jamur Colletotrichum gloeosporioides...',
        'pencegahan': 'Pangkas cabang yang terlalu rimbun...',
        'pengobatan': 'Semprotkan fungisida berbahan aktif Mankozeb...'
    },
    'Bercak_Algae': {
        'nama_lain': 'Bercak Ganggang Merah / Karat Merah',
        'penjelasan': 'Disebabkan oleh alga Cephaleuros virescens...',
        'pencegahan': 'Jagalah kebersihan kebun...',
        'pengobatan': 'Aplikasikan fungisida berbahan aktif Tembaga Oksiklorida...'
    },
    'Hawar_Daun': {
        'nama_lain': 'Bercak Basah / Rhizoctonia Leaf Blight',
        'penjelasan': 'Disebabkan oleh jamur Rhizoctonia solani...',
        'pencegahan': 'Hindari penyiraman berlebihan...',
        'pengobatan': 'Semprotkan fungisida sistemik...'
    },
    'Kanker_Batang': {
        'nama_lain': 'Busuk Phytophthora / Blendok',
        'penjelasan': 'Disebabkan oleh Phytophthora palmivora...',
        'pencegahan': 'Pastikan drainase tanah baik...',
        'pengobatan': 'Kupas bagian batang yang sakit lalu oleskan fungisida...'
    },
    'Penyakit_Lain': {
        'nama_lain': 'Gejala Non-Spesifik / Hama',
        'penjelasan': 'Kerusakan daun bukan disebabkan oleh 4 penyakit utama...',
        'pencegahan': 'Lakukan pemupukan berimbang...',
        'pengobatan': 'Gunakan insektisida sistemik...'
    },
    'Sehat': {
        'nama_lain': 'Daun Normal / Bebas Hama',
        'penjelasan': 'Daun durian dalam kondisi prima...',
        'pencegahan': 'Pertahankan pola pemupukan rutin...',
        'pengobatan': 'Tidak memerlukan tindakan pengobatan khusus.'
    }
}

class CompatibleDense(Dense):
    @classmethod
    def from_config(cls, config):
        config.pop('quantization_config', None)
        return super().from_config(config)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Muat model baru
MODEL_PATH = 'model/model_durian_v2.h5'
model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects={'Dense': CompatibleDense})

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0  # Rescale jika di train.py pakai rescale=1./255
    return np.expand_dims(img_array, axis=0)

@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='File tidak ditemukan!')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='Pilih gambar terlebih dahulu!')

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            image_url = f"/uploads/{file.filename}"

            processed_img = preprocess_image(filepath)
            predictions = model.predict(processed_img)[0]
            
            all_prob = [round(float(p) * 100, 2) for p in predictions]
            predicted_idx = np.argmax(predictions)
            hasil_prediksi = CLASS_NAMES[predicted_idx]
            keyakinan = all_prob[predicted_idx]

            # JIKA TERPREDIKSI KELAS NON_DURIAN ATAU KEYAKINAN TERLALU RENDAH
            if hasil_prediksi == 'Non_Durian' or keyakinan < 50.0:
                return render_template('index.html', 
                                       filename=file.filename,
                                       image_url=image_url,
                                       not_durian=True,
                                       confidence=keyakinan,
                                       labels=CLASS_NAMES,
                                       probabilities=all_prob)

            info = DISEASE_INFO.get(hasil_prediksi, {})

            return render_template('index.html', 
                                   filename=file.filename,
                                   image_url=image_url,
                                   hasil=hasil_prediksi, 
                                   confidence=keyakinan,
                                   info=info,
                                   labels=CLASS_NAMES,
                                   probabilities=all_prob)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Dictionary Informasi Penyakit & Penanganannya
DISEASE_INFO = {
    'Antraknosa': {
        'nama_lain': 'Bercak Daun Cokelat / Karat Daun',
        'penjelasan': 'Penyakit akibat jamur Colletotrichum gloeosporioides yang menyebabkan bercak cokelat kehitaman dari ujung/pinggir daun hingga daun mengering dan gugur.',
        'pencegahan': 'Pangkas cabang yang terlalu rimbun agar sirkulasi udara lancar dan kurangi kelembapan di sekitar pohon.',
        'pengobatan': 'Semprotkan fungisida berbahan aktif Mankozeb, Tembaga Hidroksida, atau Difenokonazol sesuai dosis kemasan.'
    },
    'Bercak_Algae': {
        'nama_lain': 'Bercak Ganggang Merah / Karat Merah',
        'penjelasan': 'Disebabkan oleh alga Cephaleuros virescens, ditandai dengan bercak bulat agak menimbul berwarna hijau kelabu hingga jingga kecokelatan seperti beludru.',
        'pencegahan': 'Jagalah kebersihan kebun (sanitasi), buang daun yang terinfeksi berat, dan optimalkan pencahayaan matahari.',
        'pengobatan': 'Aplikasikan fungisida/algaesida berbahan aktif Tembaga Oksiklorida (Copper Oxychloride).'
    },
    'Hawar_Daun': {
        'nama_lain': 'Bercak Basah / Rhizoctonia Leaf Blight',
        'penjelasan': 'Disebabkan oleh jamur Rhizoctonia solani, memicu bercak kebasah-basahan pada daun yang cepat meluas hingga daun terlihat seperti terbakar atau melepuh.',
        'pencegahan': 'Hindari penyiraman berlebihan pada daun di sore hari dan atur jarak tanam agar tidak terlalu rapat.',
        'pengobatan': 'Semprotkan fungisida sistemik berbahan aktif Azoksistrobin atau Kresoksim-metil.'
    },
    'Kanker_Batang': {
        'nama_lain': 'Busuk Phytophthora / Blendok',
        'penjelasan': 'Disebabkan oleh Phytophthora palmivora. Meski menyerang batang, gejalanya sering muncul berupa daun kuning, layu, dan gugur secara mendadak.',
        'pencegahan': 'Pastikan drainase tanah di sekitar perakaran baik dan tidak ada air yang menggenang saat musim hujan.',
        'pengobatan': 'Kupas bagian batang yang sakit lalu oleskan fungisida bahan aktif Asam Phosphit atau Mankozeb.'
    },
    'Penyakit_Lain': {
        'nama_lain': 'Gejala Non-Spesifik / Hama / Defisiensi Nutrisi',
        'penjelasan': 'Kerusakan daun bukan disebabkan oleh 4 penyakit utama di atas (bisa karena serangan ulat, kurang pupuk Nitrogen/Kalium, atau terbakar sinar matahari).',
        'pencegahan': 'Lakukan pemupukan berimbang (NPK + Mikro) dan cek rutin keberadaan hama fisik di balik daun.',
        'pengobatan': 'Gunakan insektisida sistemik jika ada indikasi serangan hama ulat/kutu, atau beri pupuk daun.'
    },
    'Sehat': {
        'nama_lain': 'Daun Normal / Bebas Hama',
        'penjelasan': 'Daun durian dalam kondisi prima, berwarna hijau segar tanpa tanda-tanda infeksi jamur atau alga.',
        'pencegahan': 'Pertahankan pola pemupukan rutin, penyiraman yang cukup, dan pemangkasan berkala.',
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

# Urutan ini harus sama dengan urutan folder/class saat model dilatih.
CLASS_NAMES = ['Antraknosa', 'Bercak_Algae', 'Hawar_Daun', 'Kanker_Batang', 'Penyakit_Lain', 'Sehat']

def build_and_load_model(model_path):
    print("Memuat model lengkap dari file .h5...")
    model = tf.keras.models.load_model(
        model_path,
        compile=False,
        custom_objects={'Dense': CompatibleDense}
    )
    print("Model AI Berhasil Dimuat Sempurna!")
    return model

MODEL_PATH = 'model/model_durian_mobilenetv2.h5'
model = build_and_load_model(MODEL_PATH)
if model.output_shape[-1] != len(CLASS_NAMES):
    raise ValueError(
        f'Jumlah class ({len(CLASS_NAMES)}) tidak cocok dengan output model '
        f'({model.output_shape[-1]}).'
    )

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)

    # Model sudah memiliki layer Rescaling(1/255); jangan rescale lagi di sini.
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

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
            return render_template('index.html', error='Pilih gambar daun terlebih dahulu!')

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            processed_img = preprocess_image(filepath)
            predictions = model.predict(processed_img)[0]
            
            all_prob = [round(float(p) * 100, 2) for p in predictions]
            
            predicted_idx = np.argmax(predictions)
            hasil_prediksi = CLASS_NAMES[predicted_idx]
            keyakinan = all_prob[predicted_idx]

            image_url = f"/uploads/{file.filename}"

            # Threshold dibendung pada 40%
            if keyakinan < 40.0:
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
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array, load_img # type: ignore
import numpy as np
import os

app = Flask(__name__)

model = load_model('cnn_model.h5')

latest_searches = []

if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename != '':
            file_path = os.path.join('static/uploads', file.filename)
            file.save(file_path)

            target_size = (224, 224) 
            image = load_img(file_path, target_size=target_size) 
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = image / 255.0  

            prediction = model.predict(image)
            result = 'Cat' if prediction[0][0] > 0.5 else 'Dog'

            image_url = url_for('static', filename=f'uploads/{file.filename}')
            latest_searches.insert(0, {'image_url': image_url, 'result': result})

            latest_searches[:] = latest_searches[:5]

            return render_template('output.html', image_url=image_url, result=result, latest_searches=latest_searches)

if __name__ == '__main__':
    app.run(debug=True)

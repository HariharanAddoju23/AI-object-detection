# AI-object-detection

import os
from flask import Flask, flash, request, redirect, render_template, send_from_directory, send_file
from werkzeug.utils import secure_filename
from zipfile import ZipFile
from ultralytics import YOLO

UPLOAD_FOLDER = 'D://file_uploading//uploads'
RESULT_FOLDER = 'D://file_uploading//runs//detect//predict'  # Assuming the result images are saved here
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['SECRET_KEY'] = "1234"

model = YOLO('best.pt')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('file')  # Retrieve multiple files
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Uploaded')
        return redirect('/')

@app.route('/runAI', methods=['POST'])
def runAI():
    images = os.listdir(UPLOAD_FOLDER)
    for image in images:
        img_path = os.path.join(UPLOAD_FOLDER, image)
        model.predict(img_path, save=True)
    flash('AI processing completed')
    return redirect('/')

@app.route("/download_result", methods=["GET"])
def download_result():
    # Zip the result images
    result_files = os.listdir(app.config['RESULT_FOLDER'])
    with ZipFile('D:\\file_uploading\\runs\\detect\\predict\\result_images.zip', 'w') as zipf:
        for file in result_files:
            print("found in results: ",file)
            zipf.write(os.path.join(app.config['RESULT_FOLDER'], file), file)
    
    # Serve the zipped file for download
    return send_file('result_images.zip', as_attachment=True)


if __name__ == '__main__':
    app.run()

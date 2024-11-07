# website.py

from ef import main
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Maximum upload size: 500MB

# Create the uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            # Process the video with ef.main
            main(video_path)
            return redirect(url_for('success', filename=filename))
    return render_template('upload.html')

@app.route('/success/<filename>')
def success(filename):
    output_video = 'annotated.mp4'
    return render_template('success.html', filename=filename, output_video=output_video)

@app.route('/output/<path:filename>')
def output_file(filename):
    return send_from_directory('output', filename)

if __name__ == '__main__':
    app.run(debug=True)
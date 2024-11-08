# website.py

from ef import main
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, Response
import os
import subprocess
from werkzeug.utils import secure_filename
import to_model
import json
app = Flask(__name__)
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.inference.ai.azure.com"
model_name = "AI21-Jamba-1.5-Large"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)


app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('websiteout', exist_ok=True)

def convert_video_for_web(input_video, output_video):
    ffmpeg_path = r'C:\\Users\\default.DESKTOP-7FKFEEG\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-7.1-full_build\\bin\\ffmpeg.exe'  # Replace with the actual path to ffmpeg.exe
    subprocess.run([
        ffmpeg_path,
        '-i', input_video,
        '-vcodec', 'libx264',
        '-acodec', 'aac',
        output_video
    ], check=True)

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
            
            # Process video
            main(video_path)
            #process video data
            filename='output\\final.txt'
            frames_data = to_model.parse_file(filename)
            print(f'processed {filename}')
            output_filename='websiteout/final.json'
            with open(output_filename, 'w') as json_file:
                json.dump(frames_data, json_file, indent=4)
            # squash_data=json.load(open("websiteout/final.json"))
            # response = client.complete(
            #     messages=[
            #         SystemMessage(content="You are a squash coach reading through big data. Tell me exactly what both players are doing based on the following json data."),
            #         UserMessage(content=f"{squash_data}"),
            #         UserMessage(content="How exactly do you know? Use specific data points from the data.")
            #     ],
                
            #     temperature=1.0,
            #     top_p=1.0,
            #     max_tokens=4000,
            #     model=model_name
            # )
            # analysis=response.choices[0].message.content
            # with open('websiteout/analysis.txt', 'w') as f:
            #     f.write(analysis)
            input_video = 'websiteout/annotated.mp4'
            output_video = 'websiteout/annotated_web.mp4'
            if not os.path.exists(input_video):
                print(f"Error: Input video not found at {input_video}")
                return "Input video not found", 400
            print(f"Websiteout dir contents: {os.listdir('websiteout')}")

            convert_video_for_web(input_video, output_video) 
            return redirect(url_for('success', filename=filename))
            
    return render_template('upload.html')

@app.route('/success/<filename>')
def success(filename):
    output_video = 'annotated_web.mp4'
    analysis=''
    try:
        with open('websiteout/analysis.txt', 'r') as f:
            analysis = f.read()
            print(analysis)
    except:
        analysis = 'Analysis not found'
    
    return render_template('success.html', filename=filename, output_video=output_video, analysis=analysis)

@app.route('/websiteout/<path:filename>')
def output_file(filename):
    return send_from_directory('websiteout', filename,  mimetype='video/mp4', as_attachment=False)

if __name__ == '__main__':
    app.run(debug=True)
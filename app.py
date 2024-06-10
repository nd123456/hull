# Importing required libs
from flask import Flask, render_template, request, url_for
from model import preprocess_img, predict_result
import os
from werkzeug.utils import secure_filename

# Instantiating flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'bmp', 'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Clear the upload folder at the start
if os.path.exists(app.config['UPLOAD_FOLDER']):
    for file in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
        os.remove(file_path)
else:
    os.makedirs(UPLOAD_FOLDER)


# Home route
@app.route("/")
def main():
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    try:
        if request.method == 'POST':
            files = request.files.getlist('files[]')
            predictions = []

            for file in files:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                with open(file_path, 'rb') as f:
                    file_bytes = f.read()
                    print("file size:\n\n", len(file_bytes))

                img = preprocess_img(file_path)  # Pass the file path instead
                pred = predict_result(img)
                predictions.append((filename, pred))

            print(predictions)
            return render_template("result.html", predictions=predictions)

    except Exception as e:
        error = f"File cannot be processed. Error: {str(e)}"
        return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000,debug=True)

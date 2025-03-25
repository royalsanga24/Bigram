from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from bigram import BigramModel
from neural_model import NeuralModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt'}

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    generated_names_bigram = []
    generated_names_neural = []
    
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read names from the file
            with open(filepath, 'r') as f:
                names = [line.strip() for line in f if line.strip()]
            
            # Train both models
            bigram_model = BigramModel()
            neural_model = NeuralModel()
            
            bigram_model.train(names)
            neural_model.train(names)
            
            # Generate new names
            count = int(request.form.get('count', 10))
            max_length = int(request.form.get('max_length', 10))
            generated_names_bigram = bigram_model.generate_names(count, max_length)
            generated_names_neural = neural_model.generate_names(count, max_length)
            
            return render_template('index.html', 
                                generated_names_bigram=generated_names_bigram,
                                generated_names_neural=generated_names_neural,
                                original_names=names,
                                file_uploaded=True)
    
    return render_template('index.html', 
                         generated_names_bigram=generated_names_bigram,
                         generated_names_neural=generated_names_neural,
                         file_uploaded=False)

if __name__ == '__main__':
    app.run(debug=True) 
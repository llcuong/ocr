from flask import Flask, request
from flask import render_template
import utils

app = Flask(__name__)
app.secret_key = 'document_scanner_app'

@app.route('/',methods=['GET','POST'])
def scandoc():
    
    if request.method == 'POST':
        file = request.files['image_name']
        upload_image_path = utils.save_upload_image(file)
        print('Image saved in = ',upload_image_path)
        return render_template('scanner.html')            
    
    return render_template('scanner.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/result')
def result():
    import main
    return render_template('result.html',results=main.display())

if __name__ == "__main__":
    app.run(debug=True)
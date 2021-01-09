from flask import Flask, render_template, request 
from werkzeug import secure_filename 
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import tensorflow as tf 
import numpy as np 
import os    
import tempfile
tempdirectory = tempfile.gettempdir()
# define the name of the directory to be created

# try:
#     os.mkdir(path)
# except OSError:
#     print ("Creation of the directory %s failed" % path)
# else:
#     print ("Successfully created the directory %s " % path)
# app.config["IMAGE_UPLOADS"] = "/mnt/c/wsl/projects/pythonise/tutorials/flask_series/app/app/static/img/uploads"
 
model = tf.keras.models.load_model('SkinCD') 
app = Flask(__name__) 
# os.makedir('/images')

  
@app.route('/') 
def upload_f(): 
    return render_template('upload.html') 
  
def finds(sfname): 
    img = image.load_img(sfname, target_size=(220, 220))

    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batch)
    print(prediction) 
    return prediction

    # return ''

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS`
  
@app.route('/uploader', methods = ['GET', 'POST']) 
def upload_file(): 
    if request.method == 'POST': 
        f = request.files['file']
        sfname = secure_filename(f.filename)
        loc = os.path.join(tempdirectory, sfname)
        f.save(loc)
        if os.path.isfile(loc):
            print ("File exist")
        else:
            print ("File not exist")
        # f = request.files['file']
        # # print(f, f.filename) 

        # f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))) 
        val = finds(loc) 
        return render_template('predict.html', ss = val) 

    # if request.method == 'POST':
    #     file = request.files['file']
    #     if file and allowed_file(file.filename):
    #         filename = secure_filename(file.filename)
    #         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    #         file.stream.seek(0) 
    #         myfile = file.file 
    #         dataframe = pd.read_csv(myfile)
    #         return
    #     else:
    #         return "Not Allowed"  
if __name__ == '__main__': 
    app.run() 
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


model = tf.keras.models.load_model('SkinCD') 
app = Flask(__name__) 

  
@app.route('/') 
@app.route('/home')
def home(): 
    return render_template('upload.html') 
  
def finds(sfname): 
    img = image.load_img(sfname, target_size=(220, 220))

    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_batch)
    print(prediction) 
    return prediction


  
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

        val = finds(loc) 
        if val < 0.5:
            val = 'Benign'
        else:
            val = 'Malignant'
        return render_template('predict.html', ss = val) 

if __name__ == '__main__': 
    app.run() 
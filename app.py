from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import tensorflow as tf
import matplotlib as mpl
import os
from PIL import Image
from pymongo import MongoClient
import io
import pandas
import base64
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


app = Flask(__name__)

# Configure the MongoDB client
client = MongoClient("mongodb+srv://omarsaad08:5RCr7kLbTk1cwiUE@cluster0.lubh9dn.mongodb.net/tumora?retryWrites=true&w=majority&appName=Cluster0")
db = client['tumora']
braincollection = db['brains']
lungcollection = db['lungs']
alzheimercollection = db['zahaimers'] 

# Path to save uploaded images
UPLOAD_FOLDER = 'static/images'
MODELS_FOLDER = 'models/'

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#-----> AI FUNCTIONALITY (TEAM) PLEASE DON'T TOUCH THAT <-----#

def load_model(model_path):
    print("Loading Saved Model")
    model = tf.keras.models.load_model(model_path)
    return model

# Function to get image array
def get_img_array(img_path, size=(224, 224)):
    img = tf.keras.utils.load_img(img_path, target_size=size)
    array = tf.keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

# Function to make Grad-CAM heatmap
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Function to decode predictions
def decode_predictions(preds):
    classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    prediction = classes[np.argmax(preds)]
    return prediction

# Function to save and display Grad-CAM
def save_and_display_gradcam(img_path, heatmap, cam_path="static/segmentation/cam.jpg", alpha=0.4):
    img = tf.keras.utils.load_img(img_path)
    img = tf.keras.utils.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)

# Function to make prediction and generate Grad-CAM
def make_prediction(img_path, model, last_conv_layer_name="Top_Conv_Layer", cam_path="static/segmentation/cam.jpg"):
    img_array = get_img_array(img_path, size=(224, 224))
    preds = model.predict(img_array)
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    save_and_display_gradcam(img_path, heatmap, cam_path=cam_path)
    return [cam_path, decode_predictions(preds)]
# Loading the Medical Model



# ---------------------------------------------------------- LungDisease Functionality -----------------------------------------------

def preprocess_image_with_generator(image_path, datagen, img_size=(200, 200)):
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create batch of size 1
    img_array = datagen.standardize(img_array)  # Apply data generator preprocessing (e.g., normalization)
    
    return img_array

def predict_single_image_with_generator(model, image_path, img_size=(200, 200)):
    datagen = ImageDataGenerator(rescale=1./255)
    classes = ['Lung Opacity', 'Normal', 'Viral Pneumonia']
    processed_image = preprocess_image_with_generator(image_path, datagen, img_size)
    prediction = model.predict(processed_image)
    predicted_class_index = np.argmax(prediction, axis=-1)[0]  # Get the class index
    predicted_class = classes[predicted_class_index]
    return predicted_class 


# ---------------------------------------------------------- LungDisease Functionality ------------------------------------------------------

# ---------------------------------------------------------- Alzheimers Functionality -----------------------------------------------
def create_gens(train_df, valid_df, test_df, batch_size):

    img_size = (224, 224)
    channels = 3 # either BGR or Grayscale
    color = 'rgb'
    img_shape = (img_size[0], img_size[1], channels)

    ts_length = len(test_df)
    test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80]))
    test_steps = ts_length // test_batch_size

    def scalar(img):
        return img

    tr_gen = ImageDataGenerator(preprocessing_function=scalar, horizontal_flip=True)
    ts_gen = ImageDataGenerator(preprocessing_function=scalar)

    train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                        color_mode=color, shuffle=True, batch_size=batch_size)

    valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                        color_mode=color, shuffle=True, batch_size=batch_size)

    # Note: we will use custom test_batch_size, and make shuffle= false
    test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size, class_mode='categorical',
                                        color_mode=color, shuffle=False, batch_size=test_batch_size)

    return train_gen, valid_gen, test_gen

def predict_with_generator(model, img_path, batch_size=1):
    classes = ["Mild Demanted" , "Moderate Demanted" , "Non demanted" ," Very mild demanted"]
    img_df = pandas.DataFrame({'filepaths': [img_path], 'labels': [None]})
    img_size = (224, 224)
    ts_gen = ImageDataGenerator(preprocessing_function=lambda x: x)
    img_gen = ts_gen.flow_from_dataframe(img_df, x_col='filepaths', y_col=None, target_size=img_size,
                                        class_mode=None, color_mode='rgb', shuffle=False, batch_size=batch_size)
    prediction = model.predict(img_gen, steps=np.ceil(1 / batch_size))
    
    predicted_class = np.argmax(prediction, axis=1)
    
    return prediction, classes[predicted_class[0]]
# ---------------------------------------------------------- Alzheimers Functionality ------------------------------------------------------

#-----> AI FUNCTIONALITY PLEASE DON'T *TOUCH* THAT <-----#

#----> BACKEND FUNCTIONALITY <----#

heart_Disease_Model = pickle.load(open(MODELS_FOLDER + 'The_Medical_Model1.pkl', 'rb')) #-> Loading model 1
BrainTumor_Model = load_model(MODELS_FOLDER + "my_model.h5") #-> Loading model 2
LungDiseaseModel = load_model(MODELS_FOLDER + "LungDiseaseCNN-2.15.0 (4).h5") #-> Loading model 3
AlzheimerModel = load_model(MODELS_FOLDER + "BrainAlzahymers.h5") #-> Loding model 4

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


# Medical model prediction route
@app.route("/Heart-predict", methods=["POST"])
def predictHeartDisease():
    req_data = request.get_json()

    age = int(req_data['age'])
    sex = int(req_data['sex'])
    cp = int(req_data['cp'])
    trestbps = int(req_data['trestbps'])
    chol = int(req_data['chol'])
    fbs = int(req_data['fbs'])
    restecg = int(req_data['restecg'])
    thalach = int(req_data['thalach'])
    exang = int(req_data['exang'])
    oldpeak = float(req_data['oldpeak'])
    slope = int(req_data['slope'])
    ca = int(req_data['ca'])
    thal = int(req_data['thal'])

    makeprediction = heart_Disease_Model.predict([[age, sex, cp, trestbps,
                                                chol, fbs, restecg,
                                                thalach, exang, oldpeak,
                                                slope, ca, thal]])
    makeprediction_prob = heart_Disease_Model.predict_proba([[age, sex, cp, trestbps,
                                                            chol, fbs, restecg,
                                                            thalach, exang, oldpeak,
                                                            slope, ca, thal]])

    output = makeprediction.tolist()
    output_2 = makeprediction_prob.tolist()

    result = "positive" if output[0] == 1 else "negative"
    return jsonify({
        "Your result is": result,
        "No_probability": output_2[0][0],
        "Yes_probability": output_2[0][1]
    })

@app.route("/BrainTumor", methods=['POST'])
def predictBrainTumor():
    direct = False
    data = request.get_json()
    direct = data.get('direct')

    #getting the image from mongodb
    if direct == False:
        request_data = request.get_json()
        patient_id = request_data.get('id')
        ray_date = request_data.get('imageDate')

        #Find the patient 
        patient_document = braincollection.find_one({'id': patient_id})

        if not patient_document:
            return jsonify({"error": "Patient not found"}), 404
        
        # Extract the image data from the rays
        ray_image_data = None
        for ray in patient_document['rays']:
            if ray.get('imageDate') == ray_date:
                ray_image_data = ray['imageData']
                ray_image_name = ray['imageName']
                break

        if not ray_image_data:
            return jsonify({"error": "Ray image data not found"}), 404

        #creating the file_path in which the photo will be saved
        img_path = os.path.join(UPLOAD_FOLDER, ray_image_name + '.png')
        
        # Convert the binary data to an image
        image = Image.open(io.BytesIO(ray_image_data))

        #saving the image in the file_path
        image.save(img_path, format='PNG')

        #predicting
        cam_path, prediction = make_prediction(img_path, BrainTumor_Model)
        segmented_image = Image.open(cam_path)

        # Convert to binary data 
        buffer = io.BytesIO()
        segmented_image.save(buffer, format='JPEG')  
        binary_data = buffer.getvalue()

        segmented_imagename = f"segmentation_{ray_image_name}"

        # Store in Mongo
        braincollection.update_one(
        {'id': patient_id, 'rays.imageDate': ray_date},
        {'$set': {'segmentation': {'imageData': binary_data, 'imageDate': ray_date, 'imageName': segmented_imagename, 'result': prediction}}}
        )

        return jsonify({"result": prediction, "image_path": img_path, "segmented_image_path": cam_path}) 
    
    #getting the image from request
    else:
        image = request.files('my_image')

        #create the image path
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)

        #saving the image in the file_path
        image.save(image_path, format='PNG')

        #predicting
        cam_path, prediction = make_prediction(image_path, BrainTumor_Model)

        with open(cam_path, "rb") as segmented_image_file:
        # Encode the binary data to Base64
            encoded_segmented_image = base64.b64encode(segmented_image_file.read()).decode('utf-8')

        return jsonify({"result": prediction, "segmented_image": encoded_segmented_image}) 



@app.route("/Alzheimer", methods=['POST'])
def predictAlzheimer():
    direct = False
    data = request.get_json()
    direct = data.get('direct')

    #getting the image from mongodb
    if direct == False:

        request_data = request.get_json()
        patient_id = request_data.get('id')
        ray_date = request_data.get('imageDate')

        #Find the patient   
        patient_Document = alzheimercollection.find_one({'id' : patient_id})

        if not patient_Document:
            return jsonify({"error": "Patient not found"}), 404
        
        # Extract the image data from the rays
        ray_image_data = None
        for ray in patient_Document['rays']:
            if ray.get('imageDate') == ray_date:
                ray_image_data = ray['imageData']
                ray_image_name = ray['imageName']
                break

        if not ray_image_data:
            return jsonify({"error": "Ray image data not found"}), 404

        #creating the file_path in which the photo will be saved
        file_path = os.path.join(UPLOAD_FOLDER, ray_image_name + '.png')
        
        # Convert the binary data to an image
        image = Image.open(io.BytesIO(ray_image_data))

        #saving the image in the file_path
        image.save(file_path, format='PNG')

        #predicting
        prediction, predicted_class = predict_with_generator(AlzheimerModel, file_path)


        # store in mongo
        alzheimercollection.update_one(
            {"id": patient_id, 'rays.imageDate': ray_date},
            {'$set': {'rays.$.result': prediction}}
        )
        
        return jsonify({"prediction": prediction, "predicted_class": predicted_class})
    
    #getting the image from request
    else:
        image = request.files('my_image')

        #create the image path
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)

        #saving the image in the file_path
        image.save(image_path, format='PNG')

        #predicting
        prediction, predicted_class = predict_with_generator(AlzheimerModel, image_path)

        return jsonify({"result": prediction, "predicted_class": predicted_class}) 




@app.route("/LungDisease", methods=['POST'])
def predictLungDisease():
    direct = False
    data = request.get_json()
    direct = data.get('direct')

    #getting the image from mongodb
    if direct == False:

        request_data = request.get_json()
        patient_id = request_data.get('id')
        ray_date = request_data.get('imageDate')

        #Find the patient   
        patient_Document = lungcollection.find_one({'id' : patient_id})

        if patient_id not in patient_Document:
            return jsonify({"error": "Patient not found"})
        
        # Extract the image data from the rays
        ray_image_data = None
        for ray in patient_Document['rays']:
            if ray.get('imageDate') == ray_date:
                ray_image_data = ray['imageData']
                ray_image_name = ray['imageName']
                break

        if not ray_image_data:
            return jsonify({"error": "Ray image data not found"})

        #creating the file_path in which the photo will be saved
        file_path = os.path.join(UPLOAD_FOLDER, ray_image_name + '.png')
        
        # Convert the binary data to an image
        image = Image.open(io.BytesIO(ray_image_data))

        #saving the image in the file_path
        image.save(file_path, format='PNG')

        prediction = predict_single_image_with_generator(LungDiseaseModel, file_path)
        lungcollection.update_one(
            {"id": patient_id, 'rays.imageDate': ray_date},
            {'$set': {'rays.$.result': prediction}}
        )
        return jsonify({"prediction": prediction})
    
    #getting the image from request
    else:
        image = request.files('my_image')

        #create the image path
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)

        #saving the image in the file_path
        image.save(image_path, format='PNG')

        #predicting
        prediction = predict_single_image_with_generator(LungDiseaseModel, image_path)

        return jsonify({"result": prediction}) 



if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify
import numpy as np
import pickle
import tensorflow as tf
import matplotlib as mpl
import os

app = Flask(__name__)
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
#-----> AI FUNCTIONALITY PLEASE DON'T *TOUCH* THAT <-----#




#----> BACKEND FUNCTIONALITY <----#
heart_Disease_Model = pickle.load(open('The_Medical_Model1.pkl', 'rb')) #-> Loading model 1
BrainTumor_Model = load_model("my_model.h5")
# Medical model prediction route
@app.route("/Heart-predict", methods=["POST"])
def predict():
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
def get_output():
    if request.method == 'POST' and 'my_image' in request.files:
        img = request.files['my_image']
        img_path = os.path.join("static/images", img.filename)
        img.save(img_path)
        cam_path, prediction_1 = make_prediction(img_path, BrainTumor_Model)
        return {"prediction": prediction_1, "image_path": img_path, "segmented_image_path": cam_path}

if __name__ == "__main__":
    app.run(debug=True)
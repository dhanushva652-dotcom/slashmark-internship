from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model("models/cat_dog_cnn.keras")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            os.makedirs("static", exist_ok=True)

            image_path = os.path.join("static", file.filename)
            file.save(image_path)

            img = image.load_img(image_path, target_size=(150, 150))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            pred = model.predict(img_array)

            if pred[0][0] > 0.5:
                prediction = "Dog"
            else:
                prediction = "Cat"

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path
    )

if __name__ == "__main__":
    app.run(debug=True)
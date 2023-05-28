import os
import urllib
import uuid

from PIL import Image
from flask import Flask, render_template, request
from keras.models import load_model
from keras.utils import load_img, img_to_array

app = Flask("ClassificationWebApp")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, "model.hdf5"))

ALLOWED_EXT = {"jpg", "jpeg", "png", "jfif"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXT


classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def predict(filename, model):
    img = load_img(filename, target_size=(32, 32))
    img = img_to_array(img)
    img = img.reshape(1, 32, 32, 3)

    img = img.astype("float32")
    img = img / 255.0
    result = model.predict(img)

    dict_result = {}
    for i in range(10):
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[0]

    prob_result = [(prob * 100).round(2)]
    class_result = [dict_result[prob]]

    return class_result, prob_result


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/success", methods=["GET", "POST"])
def success():
    error = ""
    target_img = os.path.join(os.getcwd(), "static/images")
    if request.method == "POST":
        if request.form:
            link = request.form.get("link")
            try:
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename + ".jpg"
                img_path = os.path.join(target_img, filename)
                output = open(img_path, "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result, prob_result = predict(img_path, model)

                predictions = {
                    "class1": class_result[0],
                    "prob1": prob_result[0],
                }

            except Exception as e:
                print(str(e))
                error = (
                    "This image from this site is not accesible or "
                    "inappropriate input"
                )

            if len(error) == 0:
                return render_template(
                    "success.html", img=img, predictions=predictions
                )
            else:
                return render_template("index.html", error=error)

        elif request.files:
            file = request.files["file"]
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img, file.filename))
                img_path = os.path.join(target_img, file.filename)
                img = file.filename

                class_result, prob_result = predict(img_path, model)

                predictions = {
                    "class1": class_result[0],
                    "prob1": prob_result[0],
                }

            else:
                error = (
                    "Please upload images of jpg , jpeg and png extension only"
                )

            if len(error) == 0:
                return render_template(
                    "success.html", img=img, predictions=predictions
                )
            else:
                return render_template("index.html", error=error)

    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

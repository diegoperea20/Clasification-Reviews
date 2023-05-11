from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST" or request.method == "GET":
        return render_template("index.html")
            

@app.route("/clasification", methods=["POST"])
def clasification():  
        #--------------------------- 
        option_selected = request.form['opciones']
        if option_selected == "opcion4":
             # Cargar el modelo guardado
             model = load_model("app\modelo4.h5")
        elif option_selected == "opcion3":
             model = load_model("app\modelo3.h5")
        elif option_selected == "opcion2":
             model = load_model("app\modelo2.h5")
        else:
             model = load_model("app\modelo1.h5")
        #--------------------------- 
       # Cargar el modelo guardado
       # model = load_model("app\modelo.h5")

        # Prompt the user to enter a review for classification
        user_review = request.form['review']
        # the movie was very stupid and bad  the worsdt movie in al times  :negative
        # nice video : positive

        # Tokenize the user's review using the IMDB dataset's word index
        word_index = imdb.get_word_index()
        user_tokens = [
            word_index[word] if word in word_index else 0
            for word in user_review.split()
        ]

        # Pad the user's tokens to a maximum length of 256 words
        maxlen = 256
        user_padded = pad_sequences([user_tokens], maxlen=maxlen)

        # Use the trained model to predict the sentiment of the user's review
        prediction = model.predict(user_padded)[0][0]

        # Print the predicted sentiment
        if prediction >= 0.5:
            print(f"Prediction: {prediction}")
            print("Positive review!")
            mensaje="Positive review!"
            return render_template("index.html", prediction=prediction, mensaje=mensaje ,user_review=user_review )
        else:
            print(f"Prediction: {prediction}")
            print("Negative review.")
            mensaje="Negative review."
            return render_template("index.html", prediction=prediction, mensaje=mensaje,user_review=user_review )

if __name__ == "__main__":
    app.run(debug=True)

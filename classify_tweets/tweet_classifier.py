
from keras.preprocessing import sequence, text
import numpy as np
from keras.models import model_from_json, load_model

import pickle


def predictTweet(tweet):

    if tweet == "":
        return "Please enter the tweet!", None

    max_len = 1500
    print(tweet)

    json_file = open('lstm_model_arch.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights("lstm_model.h5")


    with open('lstm_tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)

    seq = loaded_tokenizer.texts_to_sequences([tweet])
    padded = sequence.pad_sequences(seq, maxlen=max_len)

    pred = loaded_model.predict(padded)
    classes_x = np.argmax(pred, axis=1)
    print("Please",classes_x)


    classification = "Disaster Alert !" if pred[0][0] >= 0.5 else "Not a disaster !"

    print("Tweet: ", tweet)
    print("Prediction: ", pred[0][0])
    print("Classification: ", classification )

    return tweet, classification


# tweet = "Just happened a terrible car crash"
# predictTweet(tweet)

# from transformers import PegasusForConditionalGeneration, PegasusTokenizer

import NewsSentiment
import numpy as np
from flask import Flask,request,jsonify
import dill
import transformers
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import tensorflow
import keras
from summarisation import Summarizer
from sentimentAnalysis import SentimentClassifier
from tagGeneration import TagGenerator
from newsClassification import Classifier
app = Flask(__name__)


summarizer= Summarizer()
sentimentClassifier= SentimentClassifier()
tagGenerator= TagGenerator()
classifier= Classifier()

# summarizer.summarise("When the “spirit of Mumbai” is spoken about, to signify the city’s imagined resilience especially after some major calamity, it is always about the multitude. It is assumed that the faceless common citizens, who do not have the luxury to stay inside safely, venture out due to this spirit. Payal Kapadia’s debut feature All We Imagine as Light doesn’t make such assumptions about the less privileged, rather it brings them to the centre and gives them a voice.In fact, the film begins by giving voices to these multitudes, before it settles on the three protagonists. In a sequence which reminds one of Kapadia’s documentary roots, we hear from these voices what the city means to the thousands who migrate to it from all over the country to make a living. Malayali nurses Prabha (Kani Kusruti) and Anu (Divya Prabha), and Parvaty (Chhaya Kadam), an employee at the hospital they are working in, belong to that tribe.Also Read | Politics of aesthetics: How ‘Laapataa Ladies’ got a shot at the OscarsBut the film is really not about the work that they do or their daily struggles; what shines bright instead is their interior lives, their desires, disappointments, confusions and even biases. Prabha has a weary air about her, as someone who has been in the city for quite some time, with the recurring worry about a husband who has literally forgotten about her after the first few days of marriage. The last time they spoke was a year ago, after he went for a job in Germany. Maybe he has nothing more to say, she tells her friend.Anu, in contrast, is bursting with the energy of the newfound freedom in the city and the high of her secret love with Shiaz (Hridhu Haroon). With the patriarchal mores ingrained in Prabha, she finds Anu’s ways a little too troubling, and words escape out of her. The younger one has a mind of her own and only gets more daring in her escapades, but she has her own set of confusions about her future. Parvaty, meanwhile, is facing eviction from her dwelling space of over two decades. With no papers to prove her ownership, it is easy for the builders of skyscrapers to evict her. In these varying shades and scales of adversity, the three women find things that bind them together.Also Read | ‘I get overwhelmed in crowds’: Kani Kusruti on Cannes selection for ‘All We Imagine as Light’The coming together is for no act of revolt (unless the delightful scene of throwing stones at the builder’s advertisement hoarding can be called that), but just for being there for each other. From the Mumbai nights of ceaseless activity and quiet contemplation of the protagonists in their private spaces, the film in the later half takes us to a seaside Maharashtrian village of blinding sunlight and a tender calmness. In a passage of inspired writing, Prabha gets a sense of closure through a scene that exists somewhere in the comfortable space between the real and the imaginary. Yet, one is left with a feeling of the filmmaker holding something back in the final moments, choosing to let it float away gently like a kite in the breeze, rather than let it be a soaring bird. It leaves one with a mild sense of being unsatiated.At times, the film exudes the feel of a Mumbai mood piece, with some gentle jazz to go along with it. Once in a while, the expanse of the lens widens, capturing the endless row of flats with dimly or brightly lit windows; the ever-moving suburban trains and the populace reminds us that these are not just the stories of the three, but representative of the many outsiders here. Kapadia infuses a lyrical quality to even the mundane moments, although the endlessly romanticised Mumbai rains is thoughtfully turned into a frustrating hindrance to a romantic encounter.All We Imagine as Light is as much an ode to the city as it is to its outsiders, who just can’t call it home but can’t leave it too.All We Imagine as Light is currently running in select theatres in Kerala")


@app.route('/summarize',methods=['POST'])
def predict():
    data=request.get_json(force=True)
    prediction=summarizer.summarise(data['body'])
    output=prediction[0]
    return jsonify(output)

@app.route('/sentiment',methods=['POST'])
def sentiment():
    data=request.get_json(force=True)
    sentiment=sentimentClassifier.predict_sentiment(data['body'])
    output=sentiment
    return jsonify(output)

@app.route('/tag',methods=['POST'])
def tag():
    data=request.get_json(force=True)
    tags=tagGenerator.generate_tags(data['body'])
    output=tags
    return jsonify(output)

@app.route('/classify',methods=['POST'])
def classify():
    data=request.get_json(force=True)
    category=classifier.classify(data['body'])
    output=category
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)  # Ensure debug mode is enabled to get detailed logs

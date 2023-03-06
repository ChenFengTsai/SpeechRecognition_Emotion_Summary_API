from flask import Flask, render_template, request, send_from_directory
import speech_recognition as sr
import model
import summarize

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/css/<path:path>')
def serve_css(path):
    return send_from_directory('static/css', path)

@app.route('/recognize', methods=['POST'])
def recognize():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print('Say something...')
        audio = r.listen(source, timeout=2, phrase_time_limit=20)
    try:
        text = r.recognize_google(audio)
        prediction = model.prediction(text)
        
        return render_template('index.html', text=text, emotion = prediction)
    except sr.UnknownValueError:
        return render_template('index.html', text='Sorry, I could not understand what you said')
    except sr.RequestError as e:
        return render_template('index.html', text='Sorry, my speech service is down: {}'.format(e))

@app.route("/example")
def example():
    pred_sentences = "What an excellent sequel - I, in fact, like it more than its predecessor.'Top Gun: Maverick' is fantastic, simply put. I was expecting it to be good, but it's actually much more enjoyable than I had anticipated. The callbacks to the original are expertly done, the new characters are strong/well cast, it has plenty of meaning, music is fab and the action is outstanding - the aerial stuff is sensational. The story is superb, with each high stake coming across as intended - parts even gave me slight goosebumps, which is a surprise given I'm not someone who has a connection to the 1986 film. It's all super neatly put together, I honestly came close to giving it a higher rating. Tom Cruise is brilliant as he reprises the role of Maverick, while Miles Teller comes in and gives a top performance. Jennifer Connelly is another positive, though her role does kinda feel a tiny bit forced in order to have a love interest; given Kelly McGillis' (unexplained) absence. Monica Barbaro stands out most from the fresh faces, though I actually did enjoy watching them all - which is something I thought the film may struggle with, adding new people, but it's done nicely; sure Jon Hamm and Glen Powell are a little cliché, though overall I approve. A great watch - I'd highly recommend it, though naturally would suggest watching the previous film first if you haven't already."
    example_text = pred_sentences
    summarized_text = summarize.generate_summary(example_text, top_n=5)
    prediction = model.prediction(example_text)
    
    return render_template("index.html", example_text=example_text, emotion = prediction, 
                           summarized_text=summarized_text)

if __name__ == '__main__':
    app.run(debug=True)


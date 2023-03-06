from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

label_to_num = {'admiration': 0,
 'amusement': 1,
 'anger': 2,
 'annoyance': 3,
 'approval': 4,
 'caring': 5,
 'confusion': 6,
 'curiosity': 7,
 'desire': 8,
 'disappointment': 9,
 'disapproval': 10,
 'disgust': 11,
 'embarrassment': 12,
 'excitement': 13,
 'fear': 14,
 'gratitude': 15,
 'grief': 16,
 'joy': 17,
 'love': 18,
 'nervousness': 19,
 'optimism': 20,
 'pride': 21,
 'realization': 22,
 'relief': 23,
 'remorse': 24,
 'sadness': 25,
 'surprise': 26,
 'neutral': 27}

model = tf.keras.models.load_model(
    './model_final',
    custom_objects={'TFBertForSequenceClassification': TFBertForSequenceClassification}
)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# if status.assert_existing_objects_matched():
#     print("Model restored successfully!")
# else:
#     print("Model not restored!")

def prediction(pred_sentences):
    
    tf_batch = tokenizer(pred_sentences, max_length=128, padding=True, truncation=True, return_tensors='tf')
    tf_outputs = model(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
    label = tf.argmax(tf_predictions, axis=1)
    label= label.numpy()
    key_list = list(label_to_num.keys())
    val_list = list(label_to_num.values())

    label = tf.argmax(tf_predictions, axis=1)
    label = label.numpy()
    for i in range(len(pred_sentences)):
        position = val_list.index(label[i])
        return key_list[position]
    
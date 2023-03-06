from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
import pandas as pd
import subprocess
from sklearn.model_selection import train_test_split

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

device_name = tf.test.gpu_device_name()
if len(device_name) > 0:
    print("Found GPU at: {}".format(device_name))
else:
    device_name = "/device:CPU:0"
    print("No GPU, using {}.".format(device_name))


subprocess.run(['wget', '-P', './data/full_dataset/', 'https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv'])

df = pd.read_csv('./data/full_dataset/goemotions_1.csv')
df_text = df['text']
df_emotion['text'] = df_text
df_input = df_emotion.melt(id_vars='text', var_name='label', value_name = 'target')
df_input = df_input[df_input['target']==1]
df_input = df_input.drop('target', axis=1)

labels = list(df_input['label'].unique())
label_to_num = {}
for i in range(len(labels)):
    label_to_num[labels[i]] = i
df_input['label'] = df_input['label'].map(label_to_num)


train, test = train_test_split(df_input, test_size=0.2, random_state=42)

def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN): 
    train_InputExamples = train.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)

    validation_InputExamples = test.apply(lambda x: InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this case
                                                          text_a = x[DATA_COLUMN], 
                                                          text_b = None,
                                                          label = x[LABEL_COLUMN]), axis = 1)
  
    return train_InputExamples, validation_InputExamples


  
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] # -> will hold InputFeatures to be converted later

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True, # pads to the right by default # CHECK THIS for pad_to_max_length
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
            input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


DATA_COLUMN = 'text'
LABEL_COLUMN = 'label'

train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN)

train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)

validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
validation_data = validation_data.batch(32)

with tf.device(device_name):
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=28)
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer=optimizer, 
                loss=loss, 
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

# Define the callback to show the progress bar
progbar_callback = tf.keras.callbacks.ProgbarLogger(count_mode='steps')

# Define the checkpoint object and checkpoint manager
model_path = './checkpoints'
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
manager = tf.train.CheckpointManager(checkpoint, model_path, max_to_keep=5)

if __name__ == "__main__":
    for epoch in range(3):
        for batch_index, batch in enumerate(train_data):
            #print(batch[1].keys())
            loss, accuracy = model.train_on_batch(batch[0], batch[1])

            # show progress
            progbar_callback.on_train_batch_end(batch_index, {'loss': loss, 'accuracy': accuracy})

            if optimizer.iterations % 1000 == 0:
                manager.save()
        model.evaluate(validation_data, callbacks = [progbar_callback])
    checkpoint.save(model_path)


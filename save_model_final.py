from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
import tensorflow as tf

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model_path = './checkpoints'
checkpoint = tf.train.Checkpoint(model=model)
latest_checkpoint = tf.train.latest_checkpoint(model_path)
checkpoint.restore(latest_checkpoint)

if __name__ == "__main__":
    tf.keras.models.save_model(
        model,
        './model_final',
        save_format='tf',
        include_optimizer=True
    )
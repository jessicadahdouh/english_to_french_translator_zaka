import pandas as pd
import numpy as np
import string
import nltk
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


class CleanDataSet:
    def __init__(self):
        self.input_df = None
        self.output_df = None
        self.df = None

    @staticmethod
    def read_dataset(file_path):
        return pd.read_csv(file_path, header=None)

    def set_input_dataset(self, df):
        self.input_df = df

    def get_input_dataset(self):
        return self.input_df

    def set_output_dataset(self, df):
        self.output_df = df

    def get_output_dataset(self):
        return self.output_df

    @staticmethod
    def get_number_of_rows(dataset):
        # Check the number of sentences in the "english" DataFrame
        return dataset.shape[0]

    def concat_dfs(self, input_key, output_key):
        self.df = pd.concat([self.input_df, self.output_df], axis=1)
        self.df.columns = [input_key, output_key]

    def get_dataset(self):
        return self.df

    @staticmethod
    def get_rec_from_dataset(dataset, column_name, index):
        # Get the sentence at the index specified
        sentence = dataset.loc[index, column_name]
        return sentence

    def remove_punc(self):
        self.df = self.df.applymap(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        return self.df

    def add_column_length(self, column_name, column_length_name):
        self.df[column_length_name] = self.df[column_name].apply(lambda x: len(x.split()))
        return self.df

    def get_max_column_length(self, column_length_name):
        return self.df[column_length_name].max()


class TokenizeDataset:
    def __init__(self, df):
        self.df = df

    def tokenize_text(self, target_column, tokenized_column_name, language="english"):
        """
        language is the full language name (checkout nltk)
        """
        nltk.download('punkt')
        self.df[tokenized_column_name] = self.df[target_column].apply(
            lambda x: nltk.word_tokenize(x, language=language))

    def get_dataset(self):
        return self.df

    def count_unique_words(self, tokenized_column_name):
        return len(set([word for tokens in self.df[tokenized_column_name] for word in tokens]))

    def get_fit_tokenizer(self, column_to_fit):
        fit_tokenizer = Tokenizer()
        fit_tokenizer.fit_on_texts(self.df[column_to_fit])
        return fit_tokenizer

    @staticmethod
    def save_tokenizer(tokenizer, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(tokenizer, f)

    @staticmethod
    def load_tokenizer(file_path):
        with open(file_path, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer

    def text_to_sequence(self, tokenizer, seq_column_name, column_name):
        # Convert tokenized sequences to integer sequences
        self.df[seq_column_name] = tokenizer.texts_to_sequences(self.df[column_name])

    def pad_sequence(self, max_sequence_length, seq_column_name, padded_column_name):
        sequences = pad_sequences(self.df[seq_column_name], maxlen=max_sequence_length, padding='post')
        self.df[padded_column_name] = sequences.tolist()


class TrainModelTranslator:
    def __init__(self, df):
        self.df = df
        self.input_seq_length = None
        self.output_seq_length = None
        self.input_vocab_size = None
        self.output_vocab_size = None
        self.model_summary = None
        self.model = None
        self.X = None
        self.y = None
        self.X_train = None
        self.y_train = None
        self.X_validate = None
        self.y_validate = None
        self.X_test = None
        self.y_test = None

    def set_x_y(self, X, y):
        self.X = X
        self.y = y

    def set_test_train_val_sets(self, test_size, val_size):
        # Split the data into training and temporary sets
        X_train_temp, self.X_test, y_train_temp, self.y_test = train_test_split(self.X, self.y, test_size=test_size,
                                                                                random_state=42)

        # Split the temporary sets into training and validation sets
        self.X_train, self.X_validate, self.y_train, self.y_validate = train_test_split(X_train_temp, y_train_temp,
                                                                                        test_size=val_size,
                                                                                        random_state=42)

    def get_test_set(self):
        return self.X_test, self.y_test

    def get_train_set(self):
        return self.X_train, self.y_train

    def get_validation_set(self):
        return self.X_validate, self.y_validate

    @staticmethod
    def early_stopping():
        return EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    def fit_model(self, epochs=1, early_stopping=True):
        if early_stopping:
            self.model.fit(self.X_train, self.y_train, validation_data=(self.X_validate, self.y_validate),
                           epochs=epochs,
                           batch_size=32, callbacks=[self.early_stopping()])
        else:
            self.model.fit(self.X_train, self.y_train, validation_data=(self.X_validate, self.y_validate),
                           epochs=epochs,
                           batch_size=32)

        return self.model

    def lstm_model(self, input_seq_length, output_seq_length, input_vocab_size, output_vocab_size):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(input_vocab_size, 128, input_length=input_seq_length))
        model.add(tf.keras.layers.LSTM(64))
        model.add(tf.keras.layers.RepeatVector(output_seq_length))
        model.add(tf.keras.layers.LSTM(32, return_sequences=True))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_vocab_size, activation='softmax')))

        self.model = model
        # Compile the model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        self.model_summary = model.summary()

        return self.fit_model()

    def bidirectional_model(self, input_seq_length, output_seq_length, input_vocab_size, output_vocab_size):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(input_vocab_size, 128, input_length=input_seq_length))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
        model.add(tf.keras.layers.RepeatVector(output_seq_length))
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)))
        model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_seq_length, activation='softmax')))

        self.model = model

        # Compile the new model
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        self.model_summary = model.summary()

        return self.fit_model()

    def model_predict(self):
        return self.model.predict(self.X_test)

    def model_metrics(self, predictions):
        predicted_sequences = np.argmax(predictions, axis=-1)

        # Convert the true sequences and predicted sequences to 1D arrays
        y_true = self.y_test.flatten()
        y_pred = predicted_sequences.flatten()

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Calculate recall
        recall = recall_score(y_true, y_pred, average='weighted')

        # Calculate F1-score
        f1 = f1_score(y_true, y_pred, average='weighted')

        # Calculate precision
        precision = precision_score(y_true, y_pred, average='weighted')

        # Print the metrics
        print("Accuracy:", accuracy)
        print("Recall:", recall)
        print("F1-score:", f1)
        print("Precision:", precision)

    def choose_model(self, input_seq_length, output_seq_length, input_vocab_size, output_vocab_size, model_name):
        models = {"LSTM", "Bidirectional"}
        if model_name not in models:
            raise ValueError("model not available: model must be one of %r." % models)

        if model_name == "LSTM":
            model = self.lstm_model(input_seq_length, output_seq_length, input_vocab_size, output_vocab_size)
        else:
            model = self.bidirectional_model(input_seq_length, output_seq_length, input_vocab_size, output_vocab_size)

        return model





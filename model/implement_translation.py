from keras.utils import pad_sequences

from en_to_fr_model import *

# can be later added to a config.yaml file
english_dataset_file_path = r'machine_learning_certification\Challenge 7\en.csv'
french_dataset_file_path = r'machine_learning_certification\Challenge 7\fr.csv'
max_sequence_length = 14
tokenizer_file_path_en = 'english_tokenizer.pkl'
tokenizer_file_path_fr = 'french_tokenizer.pkl'

class EnToFrTranslator:
    def __init__(self, input_sentence, model):
        self.input_sentence = input_sentence
        self.model = model
        self.clean_dataset = CleanDataSet()
        self.tokenizer = None
        self.model_class = None
        self.english_tokenizer = None
        self.french_tokenizer = None
        self.df = None

    def cleaned_dataset(self):
        english = self.clean_dataset.read_dataset(english_dataset_file_path)
        french = self.clean_dataset.read_dataset(french_dataset_file_path)

        self.clean_dataset.set_input_dataset(english)
        self.clean_dataset.set_output_dataset(french)

        # concatination
        self.clean_dataset.concat_dfs("English", "French")

        self.clean_dataset.remove_punc()

        self.clean_dataset.add_column_length('English', 'ENG Length')
        self.clean_dataset.add_column_length('French', 'FR Length')

        return self.clean_dataset.get_dataset()

    def tokenized_dataset(self, df):
        self.tokenizer = TokenizeDataset(df)

        # Tokenize the sentences
        self.tokenizer.tokenize_text(target_column='English', tokenized_column_name='English Tokens',
                                     language="english")
        self.tokenizer.tokenize_text(target_column='French', tokenized_column_name='French Tokens', language="french")

        # Create a tokenizer for each sequences
        self.english_tokenizer = self.tokenizer.get_fit_tokenizer('English Tokens')
        self.french_tokenizer = self.tokenizer.get_fit_tokenizer('French Tokens')

        # Convert tokenized sequences to integer sequences
        self.tokenizer.text_to_sequence(tokenizer=self.english_tokenizer, seq_column_name='English Integer Sequences',
                                        column_name='English Tokens')
        self.tokenizer.text_to_sequence(tokenizer=self.french_tokenizer, seq_column_name='French Integer Sequences',
                                        column_name='French Tokens')

        self.tokenizer.save_tokenizer(self.english_tokenizer, tokenizer_file_path_en)
        self.tokenizer.save_tokenizer(self.french_tokenizer, tokenizer_file_path_fr)

        self.tokenizer.pad_sequence(max_sequence_length=max_sequence_length,
                                    seq_column_name='English Integer Sequences', padded_column_name='English Padded')
        self.tokenizer.pad_sequence(max_sequence_length=max_sequence_length, seq_column_name='French Integer Sequences',
                                    padded_column_name='French Padded')

        self.df = self.tokenizer.get_dataset()

    def prepare_dataset(self):
        df = self.cleaned_dataset()
        self.tokenized_dataset(df)

    def split_dataset(self, test_size=0.1, val_size=0.1):
        # train and test the dataset here
        self.model_class = TrainModelTranslator(self.df)
        X = np.array(self.df['English Padded'].tolist())
        y = np.array(self.df['French Padded'].tolist())
        self.model_class.set_x_y(X=X, y=y)
        self.model_class.set_test_train_val_sets(test_size=test_size, val_size=val_size)

    def train_test_model(self, model_name):
        english_vocab_size = len(self.english_tokenizer.word_index) + 1
        french_vocab_size = len(self.french_tokenizer.word_index) + 1
        model = self.model_class.choose_model(input_seq_length=14, output_seq_length=14,
                                              input_vocab_size=english_vocab_size, output_vocab_size=french_vocab_size,
                                              model_name=model_name)
        return model

    def test_model(self):
        # predict the result here
        return self.model.model_predict(self.X_test)

    def model_predict(self):
        return self.model.predict(self.input_sentence)


def train_model(input_sentence, model):
    translator = EnToFrTranslator(input_sentence=input_sentence, model=model)
    translator.prepare_dataset()
    translator.split_dataset()
    translator.train_test_model(model_name="LSTM")
    translator.test_model()
    return translator.model

def load_tokenizer(file_path):
    with open(file_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer


# # use the classes here - heke we call this function and send only the input sentence and the model
def translate_text(input_sentence, model):
    train_model(input_sentence, model)
    # Tokenize the input sentence
    tokenizer_en = load_tokenizer(tokenizer_file_path_en)
    tokenizer_fr = load_tokenizer(tokenizer_file_path_fr)

    input_sequence = tokenizer_en.texts_to_sequences([input_sentence])
    input_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='post')
#
#
#     # Predict the output sequence
    output_sequence = model.predict(input_sequence)
    output_sequence = np.argmax(output_sequence, axis=-1)
#
#
    output_sentence = " ".join(tokenizer_fr.index_word[token] for token in output_sequence[0] if token != 0)
#
    return output_sentence.strip()
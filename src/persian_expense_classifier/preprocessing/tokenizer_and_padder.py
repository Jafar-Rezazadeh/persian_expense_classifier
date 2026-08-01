from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import data_utils
import numpy as np
import numpy.typing as npt


class CustomTokenizerAndPadded:
    def __init__(self, num_words, maxLen, texts):
        self.num_words = num_words
        self.maxLen = maxLen
        self.texts = texts
        self.tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(texts)
        print("number Of words in vocabulary:" + str(len(self.tokenizer.word_counts)))

    def convertTextsToPaddedSequences(self, texts) -> npt.NDArray[np.str_]:

        seq = self.tokenizer.texts_to_sequences(texts)
        padded_seq = data_utils.pad_sequences(
            seq, maxlen=self.maxLen, padding="post", truncating="post"
        )
        return padded_seq

    def saveTokenizerAsJson(self):
        tokenizer_json = self.tokenizer.to_json()

        with open("build/tokenizer.json", "w", encoding="utf-8") as f:
            f.write(tokenizer_json)
            f.close()

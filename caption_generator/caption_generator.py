# coding=utf-8

import numpy as np
import pandas as pd
from keras.models import Model

from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Input, merge
# https://stackoverflow.com/questions/46397258/how-to-merge-sequential-models-in-keras-2-0
from keras.layers.merge import concatenate

from keras.preprocessing import image, sequence
import pickle
import os
from gensim.corpora import Dictionary


EMBEDDING_DIM = 128

CUR_DIR = os.path.dirname(__file__)


class CaptionGenerator(object):

    def __init__(self, data_map_path=os.path.join(CUR_DIR, '../Flickr8k_text/flickr_8k_train_dataset.txt'), init=True):
        self.dictionary = None
        self.max_cap_len = None
        self._data_map_path = data_map_path
        self.total_samples = None
        self.encoded_images = None
        if init:
            self.variable_initializer(self._data_map_path)

    @staticmethod
    def load(model_path):
        with open(model_path,'rb') as f:
            return pickle.load(f)

    def save(self, model_path):
        cg = CaptionGenerator(init=False)
        for attr in ['dictionary', 'max_cap_len']:
            setattr(cg, attr, getattr(self, attr))
        with open(model_path, 'wb') as f:
            pickle.dump(cg, f)

    def variable_initializer(self, data_map_path):

        self.dictionary = Dictionary()

        self.encoded_images = pickle.load(open(os.path.join(CUR_DIR, 'encoded_images.p'), "rb"))

        df = pd.read_csv(data_map_path, encoding="gbk", delimiter='\t')
        caps = []
        for row in df.iterrows():
            caps.append(row[1][1])
        words = [txt.split() for txt in caps]
        self.total_samples = len(caps)
        self.dictionary.add_documents(words)

        self.max_cap_len = max(len(w) for w in words)

        print("Vocabulary size: " + str(len(self.dictionary)))
        print("Maximum caption length: " + str(self.max_cap_len))
        print("Variables initialization done!")

    def data_generator(self, batch_size=32):
        partial_caps = []
        next_words = []
        images = []
        print("Generating data...")
        gen_count = 0
        df = pd.read_csv(os.path.join(CUR_DIR, '../Flickr8k_text/flickr_8k_train_dataset.txt'), encoding='gbk',
                         delimiter='\t')
        caps = []
        imgs = []
        for i in df.iterrows():
            caps.append(i[1][1])
            imgs.append(i[1][0])

        total_count = 0
        while 1:
            image_counter = -1
            for text in caps:
                image_counter += 1
                current_image = self.encoded_images[imgs[image_counter]]
                for i in range(len(text.split()) - 1):
                    total_count += 1
                    partial = [self.dictionary.token2id[txt] for txt in text.split()[:i + 1]]
                    partial_caps.append(partial)
                    next_ = np.zeros(len(self.dictionary))
                    next_[self.dictionary.token2id[text.split()[i + 1]]] = 1
                    next_words.append(next_)
                    images.append(current_image)

                    if total_count >= batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=self.max_cap_len, padding='post')
                        total_count = 0
                        gen_count += 1
                        print("yielding count: " + str(gen_count))
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []

    def load_image(self, path):
        img = image.load_img(path, target_size=(224, 224))
        x = image.img_to_array(img)
        return np.asarray(x)

    def create_model(self, ret_model=False):
        image_input = Input(shape=(4096,), name='image_input')
        x = Dense(EMBEDDING_DIM, activation='relu')(image_input)
        x = RepeatVector(self.max_cap_len)(x)

        lang_input = Input(shape=(len(self.dictionary),), dtype='int32', name='lang_input')
        x_1 = Embedding(len(self.dictionary), 256, input_length=self.max_cap_len)(lang_input)
        x_1 = LSTM(256, return_sequences=True)(x_1)
        x_1 = TimeDistributed(Dense(EMBEDDING_DIM))(x_1)

        x_all = merge.concatenate([x, x_1])
        x_all = LSTM(1000, return_sequences=False)(x_all)
        preds = Dense(len(self.dictionary), activation='softmax')(x_all)

        model = Model(inputs=[image_input, lang_input], outputs=preds)
        return model

    def get_word(self, index):
        return self.dictionary.id2token[index]

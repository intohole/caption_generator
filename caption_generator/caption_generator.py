#coding=utf-8

import numpy as np
import pandas as pd
from keras.models import Sequential,Model



from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation,Input,merge
# https://stackoverflow.com/questions/46397258/how-to-merge-sequential-models-in-keras-2-0
from keras.layers.merge import concatenate


from keras.preprocessing import image, sequence
import pickle

EMBEDDING_DIM = 128
import os

CUR_DIR = os.path.dirname(__file__)

class CaptionGenerator():

    def __init__(self):
        self.max_cap_len = None
        self.vocab_size = None
        self.index_word = None
        self.word_index = None
        self.total_samples = None
        self.encoded_images = pickle.load( open( os.path.join(CUR_DIR,'encoded_images.p'),"rb" ))
        self.variable_initializer()

    def variable_initializer(self):
        df = pd.read_csv(os.path.join(CUR_DIR,'../Flickr8k_text/flickr_8k_train_dataset.txt'),encoding="gbk", delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        for i in range(nb_samples):
            x = next(iter)
            caps.append(x[1][1])

        self.total_samples=0
        for text in caps:
            self.total_samples+=len(text.split())-1
        print ("Total samples : "+str(self.total_samples))
        
        words = [txt.split() for txt in caps]
        unique = []
        for word in words:
            unique.extend(word)

        unique = list(set(unique))
        self.vocab_size = len(unique)
        self.word_index = {}
        self.index_word = {}
        for i, word in enumerate(unique):
            self.word_index[word]=i
            self.index_word[i]=word

        max_len = 0
        for caption in caps:
            if(len(caption.split()) > max_len):
                max_len = len(caption.split())
        self.max_cap_len = max_len
        print ("Vocabulary size: "+str(self.vocab_size))
        print ("Maximum caption length: "+str(self.max_cap_len))
        print ("Variables initialization done!")


    def data_generator(self, batch_size = 32):
        partial_caps = []
        next_words = []
        images = []
        print ("Generating data...")
        gen_count = 0
        df = pd.read_csv(os.path.join(CUR_DIR,'../Flickr8k_text/flickr_8k_train_dataset.txt'),encoding='gbk', delimiter='\t')
        nb_samples = df.shape[0]
        iter = df.iterrows()
        caps = []
        imgs = []
        for i in range(nb_samples):
            x = next(iter)
            caps.append(x[1][1])
            imgs.append(x[1][0])


        total_count = 0
        while 1:
            image_counter = -1
            for text in caps:
                image_counter+=1
                current_image = self.encoded_images[imgs[image_counter]]
                for i in range(len(text.split())-1):
                    total_count+=1
                    partial = [self.word_index[txt] for txt in text.split()[:i+1]]
                    partial_caps.append(partial)
                    next_ = np.zeros(self.vocab_size)
                    next_[self.word_index[text.split()[i+1]]] = 1
                    next_words.append(next_)
                    images.append(current_image)

                    if total_count>=batch_size:
                        next_words = np.asarray(next_words)
                        images = np.asarray(images)
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=self.max_cap_len, padding='post')
                        total_count = 0
                        gen_count+=1
                        print ("yielding count: "+str(gen_count))
                        yield [[images, partial_caps], next_words]
                        partial_caps = []
                        next_words = []
                        images = []
        
    def load_image(self, path):
        img = image.load_img(path, target_size=(224,224))
        x = image.img_to_array(img)
        return np.asarray(x)


    # def create_model(self, ret_model = False):
    #     image_model = Sequential()
    #     image_model.add(Dense(EMBEDDING_DIM, input_dim = 4096, activation='relu'))
    #     image_model.add(RepeatVector(self.max_cap_len))
    #
    #     lang_model = Sequential()
    #     lang_model.add(Embedding(self.vocab_size, 256, input_length=self.max_cap_len))
    #     lang_model.add(LSTM(256,return_sequences=True))
    #     lang_model.add(TimeDistributed(Dense(EMBEDDING_DIM)))
    #
    #     model = Sequential()
    #     merge_layer = concatenate(([image_model.output,lang_model.output]))
    #     d = Dense(1, activation='softmax', name='output_layer')(merge_layer)
    #     model.add( d)
    #
    #     # model.add(Concatenate([image_model, lang_model]))
    #     model.add(LSTM(1000,return_sequences=False))
    #     model.add(Dense(self.vocab_size))
    #     model.add(Activation('softmax'))
    #
    #     print ("Model created!")
    #
    #     if(ret_model==True):
    #         return model
    #
    #     # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #     return model

    def create_model(self,ret_model = False):
        image_input = Input(shape=(4096,), name='image_input')
        x = Dense(EMBEDDING_DIM, activation='relu')(image_input)
        x = RepeatVector(self.max_cap_len)(x)

        lang_input = Input(shape=(self.max_cap_len,), dtype='int32', name='lang_input')
        x_1 = Embedding(self.vocab_size, 256, input_length=self.max_cap_len)(lang_input)
        x_1 = LSTM(256, return_sequences=True)(x_1)
        x_1 = TimeDistributed(Dense(EMBEDDING_DIM))(x_1)

        x_all = merge.concatenate([x, x_1])
        x_all = LSTM(1000, return_sequences=False)(x_all)
        preds = Dense(self.vocab_size, activation='softmax')(x_all)

        model = Model(inputs=[image_input, lang_input], outputs=preds)
        return  model

    def get_word(self,index):
        return self.index_word[index]
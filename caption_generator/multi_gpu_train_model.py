import caption_generator
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model

import os

CUR_DIR = os.path.dirname(__file__)


def train_model(weight=None, batch_size=32, epochs=10):
    cg = caption_generator.CaptionGenerator()
    model = cg.create_model()

    if weight is not None:
        model.load_weights(weight)

    file_name = 'weights-improvement-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(
        file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)

    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy'])
    parallel_model.fit_generator(
        cg.data_generator(batch_size=batch_size),
        steps_per_epoch=cg.total_samples / batch_size,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks_list)
    try:
        parallel_model.save(
            os.path.join(CUR_DIR, '../Models/WholeModel.h5'), overwrite=True)
        parallel_model.save_weights(
            os.path.join(CUR_DIR, '../Models/Weights.h5'), overwrite=True)
    except:
        print("Error in saving model.")
    print("Training complete...\n")


if __name__ == '__main__':
    train_model(epochs=50)

# import le module'tensorflow' et nomme 'tf'.
import tensorflow as tf 
# import l'API keras 
import keras as keras
# import les elements du keras comme layers,models,...
from keras import layers,models
import os
import pathlib

dataset_url="data/training/" # path to directory of our dataset
data_dir=pathlib.Path(dataset_url)

batch_size=32
image_height=180
image_width=180
train_dataset=tf.keras.preprocessing.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset="training", 
                seed=123,
                image_size=(image_height,image_width),
                batch_size=batch_size,
)
validation_dataset=tf.keras.preprocessing.image_dataset_from_directory(
                data_dir,
                validation_split=0.2,
                subset="validation",
                seed=123,
                image_size=(image_height,image_width),
                batch_size=batch_size
)
class_names=train_dataset.class_names
# number of output neurons.
num_classes=2
# Create a model for our neural network.
def create_model(): 
    model=models.Sequential([
        # Rescaling(Normalization) the values of the arrays to become between 0 and 1.(Normalization)
        layers.experimental.preprocessing.Rescaling(1/255,input_shape=(image_height,image_width,3)),
        # Convolution the image to 16 filter with same height and width. 
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        # Convolution the image to 32 filter with same height and  width.
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        # Convolution the image to 64 filter with same height and  width.
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        # Convolution the image to 128 filter with same height and  width.
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        # the input layer (couche d'entree).
        layers.Flatten(),
        # the hidden layer (couche cache) contains 128 neurons using relu as the activation function.
        layers.Dense(128,activation="relu" ,kernel_initializer='random_normal'),
        # the output layer (couche de sortie) contains 2 neurons(number of classes that we have).BufferError()
        layers.Dense(num_classes,activation=tf.nn.softmax)
    ])
    #      compile the model using 'adam' as an optimizer(learning rate) and 
    #     'sparse_categorical_crosstropy' as a type of loss for classification  multi-classes.
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )
    return model

model=create_model()

# path to directory where you will save the model, using checkpoint.
path="saved_model/train.ckpt"
# callback for stop the training once it get  the best performence ( minimum value of validation loss).
es = keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=20,restore_best_weights=True)
#  callback for save the model with its weights and biases 
callbck=keras.callbacks.ModelCheckpoint(filepath=path,save_weights=True,verbose=1,save_best_only=True)
#  callback just for write the values of loss,val_loss,accuracy..., after each epoch. 
logs=keras.callbacks.CSVLogger('my_logs.csv',separator=';',append=False)

epochs=1000
if os.path.exists(path):
    pass
else: 
    model.fit(train_dataset, validation_data=validation_dataset,epochs=epochs, callbacks=[callbck,es,logs])

# model.summary()

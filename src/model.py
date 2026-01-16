import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, TimeDistributed, Flatten
from .utils import get_vocab_lookups

def build_model(input_shape=(75, 46, 140, 1)):
    """
    Builds the LipNet architecture:
    3D CNN -> TimeDistributed Flatten -> Bi-LSTM -> Dense(Softmax)
    
    This matches the architecture used to train the 46-epoch checkpoint.
    """
    char_to_num, _ = get_vocab_lookups()
    vocab_size = char_to_num.vocabulary_size() + 1  # +1 for CTC blank
    
    model = Sequential()
    
    # 1st Conv3D Block
    model.add(Conv3D(128, 3, input_shape=input_shape, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))
    
    # 2nd Conv3D Block
    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))
    
    # 3rd Conv3D Block
    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))
    
    # TimeDistributed Flatten
    model.add(TimeDistributed(Flatten()))
    
    # Bi-LSTMs
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))
    
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))
    
    # Output Layer
    model.add(Dense(vocab_size, kernel_initializer='he_normal', activation='softmax'))
    
    return model

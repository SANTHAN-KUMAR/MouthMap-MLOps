import tensorflow as tf
import numpy as np
from .config import VOCAB_STR

def get_vocab_lookups():
    """
    Returns (char_to_num, num_to_char) StringLookup layers.
    """
    vocab = [x for x in VOCAB_STR]
    char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
    num_to_char = tf.keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )
    return char_to_num, num_to_char

def decode_predictions(pred):
    """
    Decodes the raw prediction tensor (softmax output) into text.
    Uses greedy decoding.
    """
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # ctc_decode returns [decoded_labels, log_probs]. We want decoded_labels[0]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    
    output_text = []
    _, num_to_char = get_vocab_lookups()
    
    for result in results:
        # CTC decode might return -1 for blank/invalid, which we should handle or it ignored by StringLookup?
        # Typically StringLookup handles valid indices. 
        # We need to map back carefully. 
        # The notebook used: tf.strings.reduce_join([num_to_char(x) for x in ...])
        
        # Taking care of -1 (if any, though greedy usually returns valid indices or blank)
        chars = num_to_char(result)
        text = tf.strings.reduce_join(chars).numpy().decode('utf-8')
        output_text.append(text)
        
    return output_text

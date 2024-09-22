## Written by Mr. Inal Mashukov
## Additional Transformer expert

import numpy as np
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import layers, Model
import math


class FixedPositionalEncoding(Model):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = layers.Dropout(rate=dropout)

        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.expand_dims(np.arange(0, max_len, dtype=np.float32), axis=1)
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = scale_factor * np.expand_dims(pe, axis=0)
        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x, training=True):
        x = x + self.pe[:, :tf.shape(x)[1], :]
        return self.dropout(x, training=training)
    

def get_pos_encoder(pos_encoding):
  if pos_encoding == "fixed":
    return FixedPositionalEncoding
  else:
    raise Exception("Something went wrong! ")
  
def _get_activation_fn(activation):
    if activation == 'relu':
        return tf.keras.activations.relu
    elif activation == 'gelu':
        return tf.keras.activations.gelu
    else:
        raise ValueError(f"Unknown activation function: {activation}")
    


class TransformerBatchNormEncoderLayer(Model):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model, dropout=dropout)
        self.linear1 = layers.Dense(dim_feedforward)
        self.dropout = layers.Dropout(dropout)
        self.linear2 = layers.Dense(d_model)

        self.norm1 = layers.BatchNormalization(axis= -1, epsilon=1e-5)
        self.norm2 = layers.BatchNormalization(axis= -1, epsilon=1e-5)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def call(self, src, src_mask=None, src_key_padding_mask=None):
        # Apply self-attention
        src2 = self.self_attn(src, src, src, attention_mask=src_mask)
        src = src + self.dropout1(src2)

        # Batch normalization
        #src = tf.transpose(src, perm=[1, 2, 0])  # (batch_size, d_model, seq_len)
        src = self.norm1(src)
        #src = tf.transpose(src, perm=[2, 0, 1])  # (seq_len, batch_size, d_model)

        # Feedforward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)

        # Batch normalization
        #src = tf.transpose(src, perm=[1, 2, 0])  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        #src = tf.transpose(src, perm=[2, 0, 1])  # (seq_len, batch_size, d_model)

        return src
    
class TSTransformerEncoderClassiregressor(Model):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout, pos_encoding="fixed", activation='gelu', norm="BatchNorm", freeze=False):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = layers.Dense(d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)

        # Modify the TransformerBatchNormEncoderLayer to use axis=1 for Batch Normalization
        self.transformer_encoder = [TransformerBatchNormEncoderLayer(self.d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze), activation=activation) for _ in range(num_layers)]

        self.dropout1 = layers.Dropout(dropout)

        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, max_len, self.num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        return layers.Dense(num_classes, activation='softmax')

    def call(self, X, padding_masks=None):
        inp = self.project_inp(X) * tf.sqrt(float(self.d_model))
        inp = self.pos_enc(inp)
        for layer in self.transformer_encoder:
            inp = layer(inp, src_key_padding_mask=padding_masks)
        inp = self.dropout1(inp)

        if padding_masks is not None:
            inp = inp * tf.expand_dims(padding_masks, -1)
        inp = tf.reshape(inp, (tf.shape(inp)[0], -1))
        # print(inp.shape)
        # print('1'*100)
        output = self.output_layer(inp)
        # output = layers.Dense(self.num_classes, activation='softmax')(inp)
        # print(output.shape)
        # print(f'kdd shape: {output.shape}')
        return output



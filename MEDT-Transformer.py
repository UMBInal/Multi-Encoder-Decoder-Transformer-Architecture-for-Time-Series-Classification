## Written by Mr. Inal Mashukov
## PAKDD'2024: Multi-View Transformer for multivariable time series classification

import tensorflow as tf
import keras
from tensorflow.keras import layers, Model

class MultiViewTransformerEncoderLayer(Model):
    def __init__(self, head_size, num_heads, ff_dim, dropout, l2_reg, encoder_loop):
        super(MultiViewTransformerEncoderLayer, self).__init__()
        self.self_attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)  # head_size = d_model
        self.linear1 = layers.Dense(ff_dim, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_reg))
        self.linear2 = layers.Dense(ff_dim, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_reg))
        self.norm1 = layers.BatchNormalization(epsilon=1e-5)
        self.norm2 = layers.BatchNormalization(epsilon=1e-5)
        # self.norm2 = layers.BatchNormalization(axis= -1, epsilon=1e-5)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)

        self.encoder_loop = encoder_loop

    def call(self, src):
        src2 = self.self_attn(src,src)
        src2 = self.dropout1(src2)
        res = src + src2
    #     res = layers.LayerNormalization(epsilon=1e-6)(res) 
        res = self.norm1(res)                                      
    
        # Feed-forward network
        src = self.linear1(res)
        src = self.dropout2(src)
    
        for i in range(self.encoder_loop):
            src = self.linear2(src)
            src = self.dropout3(src)
    
        src = src + res
    #     x = layers.LayerNormalization(epsilon=1e-6)(x)
        src = self.norm2(src)
        return src


class MultiViewTransformerDecoderLayer(Model):
    def __init__(self, head_size, num_heads, ff_dim, dropout, l2_reg, decoder_loop):
        super(MultiViewTransformerDecoderLayer, self).__init__()
        self.self_attn1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)  # head_size = d_model
        self.self_attn2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)
        self.linear1 = layers.Dense(ff_dim, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_reg))
        self.linear2 = layers.Dense(ff_dim, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_reg))
        self.norm1 = layers.BatchNormalization(epsilon=1e-5)
        self.norm2 = layers.BatchNormalization(epsilon=1e-5)
        self.norm3 = layers.BatchNormalization(epsilon=1e-5)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        self.dropout3 = layers.Dropout(dropout)
        self.dropout4 = layers.Dropout(dropout)

        self.decoder_loop = decoder_loop
        # self.enc_outputs = enc_outputs

    def call(self, src, enc_outputs):
        src2 = self.self_attn1(src,src)
        src2 = self.dropout1(src2)
        res = src + src2
    #     res = layers.LayerNormalization(epsilon=1e-6)(res)
        res = self.norm1(res)
        
        src = self.self_attn2(res, enc_outputs)
        src = self.dropout2(src)
        res2 = src + res
    #     res2 = layers.LayerNormalization(epsilon=1e-6)(res2)
        res2 = self.norm2(res2)
    
        # Feed-forward network
        src = self.linear1(res2)
        src = self.dropout3(src)
    
        for i in range(self.decoder_loop):
            src = self.linear2(src)
            src = self.dropout4(src)
        src = src + res2
        src = self.norm3(src)
    
        return src



class MultiViewTransformer(Model):
    def __init__(self, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout, mlp_dropout, l2_reg, encoder_loop, decoder_loop, index_ranges,n_encoder, n_decoder,num_classes):
        super(MultiViewTransformer, self).__init__()

        self.num_transformer_blocks = num_transformer_blocks
        # self.encoder_loop = encoder_loop
        # self.decoder_loop = decoder_loop

        self.n_encoder = n_encoder
        self.n_decoder = n_decoder
        self.num_classes = num_classes

        self.cropping = [tf.keras.layers.Cropping1D(cropping=index_ranges[i]) for i in range(self.n_encoder+self.n_decoder)]
        
        # self.cropping1 = tf.keras.layers.Cropping1D(cropping=index_ranges[0])
        # self.cropping2 = tf.keras.layers.Cropping1D(cropping=index_ranges[1])
        # self.cropping3 = tf.keras.layers.Cropping1D(cropping=index_ranges[2])
        # self.cropping4 = tf.keras.layers.Cropping1D(cropping=index_ranges[3])

        self.encoder = [MultiViewTransformerEncoderLayer(head_size, num_heads, ff_dim, dropout, l2_reg, encoder_loop) for _ in range(n_encoder)]
        # self.encoder_flatten = [keras.layers.Flatten() for _ in range(self.n_encoder)]
        
        # self.encoder1 = MultiViewTransformerEncoderLayer(head_size, num_heads, ff_dim, dropout, l2_reg, encoder_loop)
        # # self.encoder1 = TransformerBatchNormEncoderLayer(head_size, num_heads, ff_dim, dropout, activation='relu')
        # self.encoder2 = MultiViewTransformerEncoderLayer(head_size, num_heads, ff_dim, dropout, l2_reg, encoder_loop)

        self.decoder = [MultiViewTransformerDecoderLayer(head_size, num_heads, ff_dim, dropout, l2_reg, decoder_loop) for _ in range(n_decoder)]
        # self.decoder = [tf.keras.Sequential([MultiViewTransformerDecoderLayer(head_size, num_heads, ff_dim, dropout, l2_reg, decoder_loop),
        #                                      keras.layers.Flatten()]) for _ in range(self.n_decoder)]
        
        
        # self.decoder1 = MultiViewTransformerDecoderLayer(head_size, num_heads, ff_dim, dropout, l2_reg, decoder_loop)
        # self.decoder2 = MultiViewTransformerDecoderLayer(head_size, num_heads, ff_dim, dropout, l2_reg, decoder_loop)

        self.flatten = [keras.layers.Flatten() for _ in range(self.n_encoder+self.n_decoder)]
        # self.flat1 = keras.layers.Flatten()
        # self.flat2 = keras.layers.Flatten()
        # self.flat3 = keras.layers.Flatten()
        # self.flat4 = keras.layers.Flatten()

        
        self.concatenate = keras.layers.Concatenate()
        # self.linear = keras.layers.Dense(mlp_units, activation="relu")
        self.mlp_dropout = keras.layers.Dropout(mlp_dropout)
        self.mlp = [keras.layers.Dense(mlp_units, activation="relu") for _ in range(3)]
        
        self.softmax = keras.layers.Dense(self.num_classes, activation="softmax")
        

    # Build the model
    def call(self, src):
        inputs = []
        for i in range(self.n_encoder):
            inputs.append(f'x{i+1}')
        for i in range(self.n_decoder):
            inputs.append(f'dec{i+1}')
            
        dic = {}       
        for i, layer in enumerate(self.cropping):
            dic[inputs[i]] = layer(src)
        
        # x1 = self.cropping1(src)
        # x2 = self.cropping2(src)
        # dec1 = self.cropping3(src)
        # dec2 = self.cropping4(src)

    # To do: num_transformer_blocks....
    # # create transformer block for encoder & decoder
    #     for _ in range(self.num_transformer_blocks):
    #         print(f'number of encoder: {self.n_encoder}')
    #         x1 = self.encoder1(x1)
    #         x2 = self.encoder2(x2)

        for i, layer in enumerate(self.encoder):
            dic[inputs[i]] = layer(dic[inputs[i]])

        for i, layer in enumerate(self.decoder):
            dic[inputs[i+self.n_encoder]] = layer(dic[inputs[i+self.n_encoder]], dic[inputs[i]])
        
        # x1 = self.encoder1(x1)
        # x2 = self.encoder2(x2)
        # dec1 = self.decoder1(dec1, x1)
        # dec2 = self.decoder2(dec2, x2)

        for i, layer in enumerate(self.flatten):
            dic[inputs[i]] = layer(dic[inputs[i]]) 
        
        # x1 = self.flat1(x1)
        # x2 = self.flat2(x2)
        # dec1 = self.flat3(dec1)
        # dec2 = self.flat4(dec2)
        
        src = dic[inputs[0]]
        for i in range(1, self.n_encoder + self.n_decoder):
            src = self.concatenate([src, dic[inputs[i]]])

        # src = self.concatenate([x1, x2, dec1, dec2])
        
        for layer in self.mlp:
            src = self.mlp_dropout(layer(src))
        
        # for _ in range(3):
        #     src = self.mlp(src)
        #     src = self.mlp_dropout(src)

    # softmax layer at the end for classification
        output = self.softmax(src)

        # print(f'multiview shape: {output.shape}')
    
        return output





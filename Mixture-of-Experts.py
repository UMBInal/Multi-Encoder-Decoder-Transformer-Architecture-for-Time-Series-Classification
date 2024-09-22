## MoE gating model
from tensorflow.keras import layers, Model
import tensorflow as tf

class MixtureOfExperts(Model):
    def __init__(self, expert1, expert2):
        super(MixtureOfExperts, self).__init__()
        self.expert1 = expert1
        self.expert2 = expert2
        # self.gate = layers.Dense(2, activation='softmax')  # Gating network to choose between 2 experts
        # self.output_layer = layers.Dense(num_classes, activation='softmax')
        self.gating = tf.keras.Sequential([layers.Dense(2048, activation = 'relu'),
                                           layers.Dense(2, activation='softmax')])

    def call(self, inputs):
        # pass the inputs through the gate:
        # print(f"Input shape: {inputs.shape}")     # (batch,features,1)      (128, 22, 1)
        gate = self.gating(tf.squeeze(inputs))  
        # print(f"Gate shape: {gate.shape}")        # (batch,# of experts)    (128, 2)
        gate = tf.expand_dims(gate,axis=2)          # (batch,# of experts,1)  (128, 2， 1)
        expert1_output = self.expert1(inputs)       
        expert2_output = self.expert2(inputs)
        # print(f"Expert1 output shape: {expert1_output.shape}")       # (batch, # of classes)   (128, 9)
        # print(f"Expert2 output shape: {expert2_output.shape}")
          
        experts_output = tf.stack([expert1_output,expert2_output],axis=1)
        # print(f"Stack output shape: {experts_output.shape}")         # (batch, # of experts, # of classes)   (128, 2, 9)
        # stack 2 experts' outputs in dimension 1, sum of each elements in dimension 2 is 1 (after softmax in expert model)
        

        combined_output = tf.reduce_sum(experts_output * gate, axis=1)
        # print(f"Combined output shape: {combined_output.shape}")       # (batch, # of classes)   (128, 9)
        # gate is gating weights after softmax, 2 experts, so there are 2 numbers for each sample, and the sum is 1. The first one is for expert1, the second one is for expert2, use these weights multiple to experts' outputs, the then sum(expert1's output, expert2's output)=1. No more softmax needed.
        
        return combined_output

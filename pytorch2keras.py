import torch
import torch.nn as nn
from torch.autograd import Variable
import keras.backend as K
from keras.models import *
from keras.layers import *

import torch
from torchvision.models import squeezenet1_1


class PytorchToKeras(object):
    def __init__(self,pModel,kModel):
        super(PytorchToKeras,self)
        self.__source_layers = []
        self.__target_layers = []
        self.pModel = pModel
        self.kModel = kModel

        K.set_learning_phase(0)

    def __retrieve_k_layers(self):

        for i,layer in enumerate(self.kModel.layers):
            if len(layer.weights) > 0:
                self.__target_layers.append(i)

    def __retrieve_p_layers(self,input_size):

        input = torch.randn(input_size)

        input = Variable(input.unsqueeze(0))

        hooks = []

        def add_hooks(module):

            def hook(module, input, output):
                if hasattr(module,"weight"):
                    self.__source_layers.append(module)

            if not isinstance(module, nn.ModuleList) and not isinstance(module,nn.Sequential) and module != self.pModel:
                hooks.append(module.register_forward_hook(hook))

        self.pModel.apply(add_hooks)


        self.pModel(input)
        for hook in hooks:
            hook.remove()

    def convert(self,input_size):
        self.__retrieve_k_layers()
        self.__retrieve_p_layers(input_size)

        for i,(source_layer,target_layer) in enumerate(zip(self.__source_layers,self.__target_layers)):

            weight_size = len(source_layer.weight.data.size())

            transpose_dims = []

            for i in range(weight_size):
                transpose_dims.append(weight_size - i - 1)

            self.kModel.layers[target_layer].set_weights([source_layer.weight.data.numpy().transpose(transpose_dims), source_layer.bias.data.numpy()])

    def save_model(self,output_file):
        self.kModel.save(output_file)
    def save_weights(self,output_file):
        self.kModel.save_weights(output_file)


"""
We explicitly redefine the Squeezent architecture since Keras has no predefined Squeezent
"""

def squeezenet_fire_module(input, input_channel_small=16, input_channel_large=64):

    channel_axis = 3

    input = Conv2D(input_channel_small, (1,1), padding="valid" )(input)
    input = Activation("relu")(input)

    input_branch_1 = Conv2D(input_channel_large, (1,1), padding="valid" )(input)
    input_branch_1 = Activation("relu")(input_branch_1)

    input_branch_2 = Conv2D(input_channel_large, (3, 3), padding="same")(input)
    input_branch_2 = Activation("relu")(input_branch_2)

    input = concatenate([input_branch_1, input_branch_2], axis=channel_axis)

    return input


def SqueezeNet(input_shape=(224,224,3)):
    image_input = Input(shape=input_shape)

    network = Conv2D(64, (3,3), strides=(2,2), padding="valid")(image_input)
    network = Activation("relu")(network)
    network = MaxPool2D( pool_size=(3,3) , strides=(2,2))(network)

    network = squeezenet_fire_module(input=network, input_channel_small=16, input_channel_large=64)
    network = squeezenet_fire_module(input=network, input_channel_small=16, input_channel_large=64)
    network = MaxPool2D(pool_size=(3,3), strides=(2,2))(network)

    network = squeezenet_fire_module(input=network, input_channel_small=32, input_channel_large=128)
    network = squeezenet_fire_module(input=network, input_channel_small=32, input_channel_large=128)
    network = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(network)

    network = squeezenet_fire_module(input=network, input_channel_small=48, input_channel_large=192)
    network = squeezenet_fire_module(input=network, input_channel_small=48, input_channel_large=192)
    network = squeezenet_fire_module(input=network, input_channel_small=64, input_channel_large=256)
    network = squeezenet_fire_module(input=network, input_channel_small=64, input_channel_large=256)

    #Remove layers like Dropout and BatchNormalization, they are only needed in training
    #network = Dropout(0.5)(network)

    network = Conv2D(1000, kernel_size=(1,1), padding="valid", name="last_conv")(network)
    network = Activation("relu")(network)

    network = GlobalAvgPool2D()(network)
    network = Activation("softmax",name="output")(network)


    input_image = image_input
    model = Model(inputs=input_image, outputs=network)

    return model

# def Crnn(input_shape):
#     # Make Networkw
#     inputs = Input(name='the_input', shape=input_shape, dtype='float32')  # (None, 128, 64, 1)
#
#     # Convolution layer (VGG)
#     inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs)  # (None, 128, 64, 64)
#     inner = Activation('relu')(inner)
#     inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)
#
#     inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, 64, 32, 128)
#     inner = Activation('relu')(inner)
#     inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 128)
#
#     inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
#     inner = BatchNormalization()(inner)
#     inner = Activation('relu')(inner)
#     inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
#     inner = Activation('relu')(inner)
#     inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 32, 8, 256)
#
#     inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  # (None, 32, 8, 512)
#     inner = BatchNormalization()(inner)
#     inner = Activation('relu')(inner)
#     inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
#     inner = Activation('relu')(inner)
#     inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, 32, 4, 512)
#
#     inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)  # (None, 32, 4, 512)
#     inner = BatchNormalization()(inner)
#     inner = Activation('relu')(inner)
#
#     # CNN to RNN
#     inner = Reshape(target_shape=((32, 2048)), name='reshape')(inner)  # (None, 32, 2048)
#     inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)
#
#     # RNN layer
#     lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)  # (None, 32, 512)
#     lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
#     lstm1_merged = add([lstm_1, lstm_1b])  # (None, 32, 512)
#     lstm1_merged = BatchNormalization()(lstm1_merged)
#     lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
#     lstm_2b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(
#         lstm1_merged)
#     lstm2_merged = concatenate([lstm_2, lstm_2b])  # (None, 32, 1024)
#     lstm_merged = BatchNormalization()(lstm2_merged)
#
#     # transforms RNN output to character activations:
#     inner = Dense(num_classes, kernel_initializer='he_normal', name='dense2')(lstm2_merged)  # (None, 32, 63)
#     y_pred = Activation('softmax', name='softmax')(inner)
#
#     labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')  # (None ,8)
#     input_length = Input(name='input_length', shape=[1], dtype='int64')  # (None, 1)
#     label_length = Input(name='label_length', shape=[1], dtype='int64')  # (None, 1)
#
#     # Keras doesn't currently support loss funcs with extra parameters
#     # so CTC loss is implemented in a lambda layer
#     loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
#         [y_pred, labels, input_length, label_length])  # (None, 1)
# #

keras_model = SqueezeNet()


#Lucky for us, PyTorch includes a predefined Squeezenet
pytorch_model = squeezenet1_1(pretrained=True)

#Load the pretrained model
# pytorch_model.load_state_dict(torch.load("squeezenet.pth"))

#Time to transfer weights

converter = PytorchToKeras(pytorch_model,keras_model)
converter.convert((3,224,224))

#Save the weights of the converted keras model for later use
converter.save_weights("squeezenet.h5")
__doc__ ="""
VGG16 in Very Deep Convolutional Networks For Large-Scale Image Recconition
All of thing from scrach VGG16 

Load Image
Pooing Layer 


"""

from pickle import TRUE
from lib import *

CWD = os.path.dirname(os.path.realpath(__file__))

tf.random.set_seed(0)

def __block_1(model)->Model:
    __summary__="""   
    Block 1 of VGG16
    """

    model.add(Conv2D(64,kernel_size=(3,3),padding= "same",
                    activation="relu", input_shape=(224,224,3)))

    model.add(Conv2D(64,kernel_size=(3,3),padding= "same",
                    activation="relu"))

    model.add (MaxPooling2D((2,2),strides=(2,2)))

    model.build()
    return model
def __block_2(model):
    __summary__="""   
    Block 1 of VGG16
    """
    model.add(Conv2D(128,kernel_size=(3,3),padding="same",activation="relu"))
    model.add(Conv2D(128,kernel_size=(3,3),padding="same",activation="relu"))
    model.add (MaxPooling2D((2,2),strides=(2,2)))
    return model

def __block_3(model):
    __summary__="""   
    Block 1 of VGG16
    """
    model.add(Conv2D(256,kernel_size=(3,3),padding="same",activation="relu"))
    model.add(Conv2D(256,kernel_size=(3,3),padding="same",activation="relu"))
    model.add(Conv2D(256,kernel_size=(3,3),padding="same",activation="relu"))
    model.add (MaxPooling2D((2,2),strides=(2,2)))
    return model   

def __block_4(model):
    __summary__="""   
    Block 1 of VGG16
    """
    model.add(Conv2D(512,kernel_size=(3,3),padding="same",activation="relu"))
    model.add(Conv2D(512,kernel_size=(3,3),padding="same",activation="relu"))
    model.add(Conv2D(512,kernel_size=(3,3),padding="same",activation="relu"))
    model.add (MaxPooling2D((2,2),strides=(2,2)))

    return model    

def __block_5(model):
    __summary__="""   
    Block 1 of VGG16
    """
    model.add(Conv2D(512,kernel_size=(3,3),padding="same",activation="relu"))
    model.add(Conv2D(512,kernel_size=(3,3),padding="same",activation="relu"))
    model.add(Conv2D(512,kernel_size=(3,3),padding="same",activation="relu"))
    model.add (MaxPooling2D((2,2),strides=(2,2)))

    return model        
def create_model(isDisplay_feature_map):

    model =keras.Sequential()
    model = __block_1(model)
    model = __block_2(model)
    model = __block_3(model)
    model = __block_4(model)
    model = __block_5(model)
    if isDisplay_feature_map :
        model.summary()
        return model
    # Top 
    model.add(Flatten())
    model.add(Dense(4096,activation="relu"))
    model.add(Dense(4096,activation="relu"))
    model.add(Dense(2,activation="softmax"))

    model.summary()
    return model
def main():
    IS_DISPLAY =True
    # Load image
    img=cv2.imread(os.path.join(CWD,r"Images\134206.jpg"))

    # resize to image base on standard size of VGG16 
    img =cv2.resize(img,(224,224))
    
    # Create model
    model = create_model(isDisplay_feature_map= IS_DISPLAY)

    # Generate 
    result = model.predict(np.array([img]))

    # Display Feature Map
    if IS_DISPLAY:
        for id in range(256):
            # Show step by step of the image feature

            feature_img = result[0,:,:,id]
            ax = plt.subplot(32,32,id+1)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_img,cmap = 'gray')
        plt.show()
    return None
if __name__ =="__main__":
    main()
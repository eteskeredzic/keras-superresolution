# Enkoder-Dekoder architecture in Keras
# Don't forget the imports

# input dimensions
w = 80
h = 80


ul = layers.Input(shape=(w, h, 3))


# encoder
l1 = layers.Conv2D(filters=64, 
                   kernel_size=3, 
                   padding='same', 
                   activation='relu')(ul)

l2 = layers.Conv2D(128, 
                   kernel_size=3, 
                   padding='same', 
                   activation='relu')(l1)

# half
l3 = layers.MaxPool2D(pool_size=2)(l2) 

l4 = layers.Conv2D(256, 
                   kernel_size=3,  
                   padding='same', 
                   activation='relu')(l3)

# decoder
l5 = layers.Conv2D(256, 
                   kernel_size=3,  
                   padding='same', 
                   activation='relu')(l4)

# enlarge
l6 = layers.UpSampling2D(size=(2, 2))(l5)

l7 = layers.Conv2D(128, 
                   kernel_size=3, 
                   padding='same',
                   activation='relu')(l6)

l8 = layers.Conv2D(filters=64, 
                   kernel_size=3, 
                   padding='same', 
                   activation='relu')(l7)

l9 = Conv2D(3, 
            kernel_size=3,  
            padding='same')(l8)

ednet = Model(inputs=ul, outputs=l9)

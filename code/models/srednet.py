# SREDNet arhitektura u Keras
# Podrazumijeva se import potrebnih klasa
# Edvin Teskeredzic, 2021

# parametri za dimenziju ulaza
w = 80
h = 80


ul = layers.Input(shape=(w, h, 3))


# enkoder dio mreze
le1 = layers.Conv2D(64, 
                   kernel_size=3, 
                   padding='same', 
                   activation='relu',
                   kernel_initializer=tf.keras.initializers.HeUniform)(ul)
                   
le2 = layers.Conv2D(64, 
                   kernel_size=3, 
                   padding='same', 
                   activation='relu',
                   kernel_initializer=tf.keras.initializers.HeUniform)(le1)
                   
le3 = layers.MaxPool2D(pool_size=2)(le2) 

le4 = layers.Conv2D(128, 
                   kernel_size=3, 
                   padding='same', 
                   activation='relu',
                   kernel_initializer=tf.keras.initializers.HeUniform)(le3)
                   
le5 = layers.Conv2D(128, 
                   kernel_size=3, 
                   padding='same', 
                   activation='relu',
                   kernel_initializer=tf.keras.initializers.HeUniform)(le4)
                   
le6 = layers.MaxPool2D(pool_size=2)(le5) 


# sloj izmedju enkodera i dekodera
l7 = layers.Conv2D(256, 
                   kernel_size=3, 
                   padding='same', 
                   activation='relu',
                   kernel_initializer=tf.keras.initializers.HeUniform)(le6)


# dekoder dio mreze
ld8 = layers.UpSampling2D(size=(2, 2))(l7)

ld9 = layers.Conv2D(128, kernel_size=3, 
                   padding='same',
                   activation='relu', 
                   kernel_initializer=tf.keras.initializers.HeUniform)(ld8)
                   
ld10 = layers.Conv2D(128, 
                   kernel_size=3,  
                   padding='same', 
                   activation='relu',
                   kernel_initializer=tf.keras.initializers.HeUniform)(ld9)  

ld11 = layers.add([le5, ld10]) # skip veza

ld12 = layers.UpSampling2D(size=(2, 2))(ld11)

ld13 = layers.Conv2D(64, 
                   kernel_size=3, 
                   padding='same',
                   activation='relu', 
                   kernel_initializer=tf.keras.initializers.HeUniform)(ld12)  
                   
ld14 = layers.Conv2D(64, 
                   kernel_size=3, 
                   padding='same', 
                   activation='relu',
                   kernel_initializer=tf.keras.initializers.HeUniform)(ld13) 

l15 = layers.add([le2, ld14]) # skip veza                   

ld16 = layers.Conv2D(3, 
             kernel_size=3, 
             padding='same', 
             kernel_initializer=tf.keras.initializers.HeUniform)(ld15)

ld17 = layers.LeakyReLU(alpha=0.2)(ld16)

rednet = Model(inputs=(ul), outputs=ld17)

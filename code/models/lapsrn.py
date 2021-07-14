# RGB MS-LapSRN arhitektura u Keras
# don't forget the imports
# Edvin Teskeredzic, 2021

# input size
w = 40
h = 40

# MS-LapSRN parameters (from paper)
d = 5
r = 8


# initial convolution layer
def c_in():

    ul = layers.Input(shape=(None, None, 3))
    
    x = layers.Conv2D(filters=64, 
                      kernel_size=(3, 3), 
                      padding='same', 
                      kernel_initializer=tf.keras.initializers.he_uniform(seed=1561))(ul)

    model = Model(inputs=ul, 
                  outputs=x, 
                  name='c_in')

    return model


# recursive block
def rekurzivni_blok(d):

    ul = layers.Input(shape=(None, None, 64))
    
    x = layers.LeakyReLU(alpha=0.2)(ul)
    
    x = layers.Conv2D(filters=64, 
                      kernel_size=(3, 3), 
                      padding='same', 
                      kernel_initializer=tf.keras.initializers.he_uniform(seed=1561))(x)

    for _ in range(d - 1):
        x = layers.LeakyReLU(alpha=0.2)(x)
        x = layers.Conv2D(filters=64, 
                          kernel_size=(3, 3), 
                          padding='same', 
                          kernel_initializer=tf.keras.initializers.he_uniform(seed=1561))(x)
        
    model = Model(inputs=ul, 
                  outputs=x, 
                  name='rekurzivni_blok')

    return model


# embedding layer (array of recursive blocks)
def niz_rekurzivni_blok(d, r):

    embedding = rekurzivni_blok(d)
    
    ul = layers.Input(shape=(None, None, 64))

    x = embedding(ul)
    x = layers.add([x, ul]) # skip connection

    for i in range(r - 1):
        x = embedding(x)
        x = layers.add([x, ul])

    x = layers.Conv2DTranspose(filters=64, 
                               kernel_size=(3, 3), 
                               strides=(2, 2), 
                               padding='same',
                               kernel_initializer=tf.keras.initializers.he_uniform(seed=1561))(x)
                               
    x = layers.LeakyReLU(alpha=0.2)(x)

    model = Model(inputs=ul, 
                  outputs=x, 
                  name='niz_rekurzivni_blok')

    return model


# residual prediction
def res_p():

    ul = layers.Input(shape=(None, None, 64))

    res = layers.Conv2D(filters=3, 
                        kernel_size=(3, 3), 
                        padding='same', 
                        kernel_initializer=tf.keras.initializers.he_uniform(seed=1561))(ul)

    model = Model(inputs=ul, 
                  outputs=res, 
                  name='res_p')

    return model


# enlargement
def uvecaj():

    ul = layers.Input(shape=(None, None, 3))

    upsample = layers.Conv2DTranspose(filters=3, 
                                      kernel_size=(4, 4), 
                                      strides=(2, 2),
                                      padding='same', 
                                      kernel_initializer=tf.keras.initializers.he_uniform(seed=1561))(ul)

    model = Model(inputs=ul, 
                  outputs=upsample, 
                  name='uvecaj')

    return model


# returns LapSRN
# if four is True, we are doing 4x enlargement
def MS_LapSRN(d, r, four=False):

    # get models
    c_in_model = c_in()
    niz_model = niz_rekurzivni_blok(d, r)
    res_model = res_p()
    uvecaj_model = uvecaj()
    
    # define entry point into main network
    ulaz = layers.Input(shape=(w, h, 3))

    # stack models on one another
    c_in_1 = c_in_model(ulaz)
    uvecaj_1 = uvecaj_model(ulaz)
    niz_1 = niz_model(c_in_1)
    res_1 = res_model(niz_1)

    # define the output
    hr_x2 = layers.add([uvecaj_1, res_1])

    # if we are doing 4x, add another layer to the pyramid
    if four == True:
      uvecaj_2 = uvecaj_model(hr_x2)  
      niz_2 = niz_model(niz_1)
      res_2 = res_model(res_1)

      hr_x4 = layers.add([uvecaj_2, res_2])

      model = Model(inputs=ulaz, outputs=[hr_x4])
      return model

    else:
      model = Model(inputs=ulaz, outputs=[hr_x2])
    return model


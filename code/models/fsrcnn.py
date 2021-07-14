# RGB FSRCNN architecture in Keras
# don't forget the imports

# input size parameters
w = 40
h = 40

# FSRCNN parameters (from paper)
d = 48
s = 16
m = 4
upscale = 2

# define sequential model
fsrcnn = Sequential()


# feature extraction
fsrcnn.add(
        tf.keras.layers.Conv2D(
            input_shape=(w, h, 3),
            filters=d,
            kernel_size=5,
            padding="same",
            kernel_initializer=tf.keras.initializers.he_normal()
        )
    )
fsrcnn.add(tf.keras.layers.PReLU(shared_axes=[1, 2]))

# shrinking
fsrcnn.add(
        tf.keras.layers.Conv2D(
            filters=s,
            kernel_size=1,
            padding="same",
            kernel_initializer=tf.keras.initializers.he_normal(),
        )
    )
fsrcnn.add(tf.keras.layers.PReLU(shared_axes=[1, 2]))


# nonlinear mapping (m layers)
for i in range(m):
  fsrcnn.add(
        tf.keras.layers.Conv2D(
            filters=s,
            kernel_size=3,
            padding="same",
            kernel_initializer=tf.keras.initializers.he_normal(),
        )
    )
  fsrcnn.add(tf.keras.layers.PReLU(shared_axes=[1, 2]))


# expand
fsrcnn.add(
        tf.keras.layers.Conv2D(
            filters=d,
            kernel_size=1,
            padding="same",
            kernel_initializer=tf.keras.initializers.he_normal(),
        )
    )
fsrcnn.add(tf.keras.layers.PReLU(shared_axes=[1, 2]))


# deconvolution
fsrcnn.add(
        tf.keras.layers.Conv2DTranspose(
            filters=3,
            kernel_size=9,
            strides=(upscale, upscale),
            padding="same",
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001),
        )
    )
    
# build model    
fsrcnn.build()


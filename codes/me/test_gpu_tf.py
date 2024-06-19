import tensorflow as tf
devices = tf.config.list_physical_devices()
print("\nDevices: ", devices)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  details = tf.config.experimental.get_device_details(gpus[0])
  print("GPU details: ", details)

cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()
with tf.device('/GPU:0'): #explicitly tells model to train using GPU via TF's device context manager (note: even w/o this, tf will automatically use GPU if it's properly configured.)
    model = tf.keras.applications.ResNet50(
        include_top=True,
        weights=None,
        input_shape=(32, 32, 3),
        classes=100,)
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
    model.compile(optimizer="Adam", loss=loss_fn, metrics=["accuracy"])
    
    model.fit(x_train, y_train, epochs=5, batch_size=128) #64 samples per batch; in a 50,000 sample datset, this means 782 batches per epoch.
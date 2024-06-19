import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #disables GPU usage entirely

import tensorflow as tf
devices = tf.config.list_physical_devices()
print("\nDevices: ", devices)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  details = tf.config.experimental.get_device_details(gpus[0])
  print("GPU details: ", details)

cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()

with tf.device('/CPU:0'): #explicitly tells model to train using GPU via TF's device context manager (note: even w/o this, tf will automatically use GPU if it's properly configured.)
    model = tf.keras.applications.ResNet50(
        include_top=True,
        weights=None,
        input_shape=(32, 32, 3),
        classes=100,)
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
    model.compile(optimizer="Adam", loss=loss_fn, metrics=["accuracy"])
    
    model.fit(x_train, y_train, epochs=5, batch_size=128) #64 samples per batch; in a 50,000 sample datset, this means 782 batches per epoch.
    #advantages of batches:
        #parallel processing - samples in a batch can be processed simultaneously
            #memory -> processing the entire sample at once would require a significant amount of memory; processing 64 samples at a time is more memory efficient.
        #mini-batch descent --> parameters updated more frequently -> can lead to faster convergence
        #noise reduction - the noise in the gradient introduced by using different batches helps generalize/normalize the model (cancels out noise in the gradient; helps prevent overfitting))


# log -> tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
    #Basically saying that the GPU does not have a PCI bus ID, which is fine for integrated GPUs like in Apple silicon.

#general speed I see using the GPU:
    # with GPU (64 batch): (~80% GPU usage on Actiivty )
        #1st epoch - 700~900ms / step; 727s total
        #2nd epoch - 2s / step; 1194s total
        #3rd epoch - 785s total
    # with CPU (64 batch): 
        #1st epoch - 700 ms /step' 607s total
        #2nd epoch - 604 s total
    
    # with GPU (128 batch): (~80% GPU usage on Actiivty )
        #1st epoch - 467s total
        #2nd epoch - 387 s total
        #3rd epoch - 401 s total
        #4th epoch - 411 s total
    # with CPU (128 batch): 
        #1st epoch - 571s total
        #2nd epoch - 551s total
        #3rd epoch - 556s total
    

    #conclusion:
        #the batch size of 64 is too small for the GPU to be able to show its full potential. Since it's a smaller size, CPU is better.
        #the batch size of 128 is better for the GPU, as it can process more samples at once.
    
#WHEN TO USE A CPU VS GPU?
	# 1.	Dataset Size:
	# •	Small Dataset:
	# •	Example: less than 10,000 samples.
	# •	Training on CPU can be sufficient.
	# •	Medium Dataset:
	# •	Example: 10,000 to 100,000 samples.
	# •	A GPU can be beneficial, especially if the model is complex.
	# •	Large Dataset:
	# •	Example: more than 100,000 samples.
	# •	A GPU is highly recommended for efficiency.
	# 2.	Model Complexity:
	# •	Simple Models:
	# •	Example: Logistic Regression, Small MLPs.
	# •	CPU may be sufficient, especially for small to medium datasets.
	# •	Complex Models:
	# •	Example: Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Transformers.
	# •	GPUs can provide significant speedups.
	# 3.	Training Time:
	# •	If the training process is taking too long on a CPU (e.g., several hours or more), switching to a GPU can drastically reduce the time.
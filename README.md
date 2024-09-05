Simple out-of-distribution detector for MNIST Digits. 

Trains an autoencoder to learn a laten space for the MNIST dataset. Will return the closest example from a small set of examples, otherwise will return OOD.

Usage:

>> python convert_paint.py /path/to/image/ # if the image is hand-drawn
>> python train.py # train model and save locally
>> python infer.py /path/to/examples /path/to/converted/image # run model inference and return representative or OOD

### MNIST classification problem

#### Download dataset

You need to first download the MNIST dataset from the official [webpage](http://yann.lecun.com/exdb/mnist/).

In particular, you need following four files:
- [train-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz):  training set images (9912422 bytes) 
- [train-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz):  training set labels (28881 bytes) 
- [t10k-images-idx3-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz):   test set images (1648877 bytes) 
- [t10k-labels-idx1-ubyte.gz](http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz):   test set labels (4542 bytes)

after you download the dataset, put them into the folder `./data`:
```bash
makir data
mv train-images-idx3-ubyte.gz data/
mv train-labels-idx1-ubyte.gz data/
mv t10k-images-idx3-ubyte.gz data/
mv t10k-labels-idx1-ubyte.gz data/
```

#### Run the script

All the training script is maintained in the `main.py` file, which you do not need to implement. You can run the script by doing:
```bash
python main.py
```
 The main task for this problem is to implement the functions we left for you in the `mlp.py`, which defines the class for `OneLayerNN`. 
 Please read the docs for each function and implement it. You need to submit your implementation in this assignment.  
 Please contact me if you have any problems for the doc string: jungao@cs.toronto.edu
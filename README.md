# Stochastic Gradient Descent

Python scripts and data files containing both linearly separable and polynomialy separable data.

# The task of training a linear classifier in 2D
The task is to train a linear classifier on the provided dataset (`linsep-traindata.csv` and `linsep-trainclass.csv`). The data are linearly separable, so a simple Perceptron algorithm will do. 

The data are stored in a following way:

* `linsep-traindata.csv` contains a list of training samples with one sample per line. Each sample is defined by pair of numbers representing values of two features `x1` and `x2`.
* `linsep-trainclass.csv` contains assignment of each sample to a specific class (`1` and `-1`), i.e., on each line, there is either `1` or `-1`. the lines in this file correspond to the lines in the `linseop-traindata.csv` file, i.e., a sample on the i-th line from the `linsep-traindata.csv` file belongs to a class difined on the i-th line in the `linsep-trainclass.csv` file.

**Advanced options**

* Implement evaluation metrics evaluating the quality of the classifier
* Extend the classifier to work on linearly non-separable data defined in `nonlinsep-traindata.csv` and `nonlinsep-trainclass.csv` (having the same structure as described above.
* Create a chart depicting the classiffier as well as the samples in 2D.

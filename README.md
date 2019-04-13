# AdaBoost

## CS 4342/5342

---
Project 2: Ensemble Method (AdaBoost with decision stumps as base classifiers)
Due: April 20, 2019
Points: 100
---

In this project you will apply the AdaBoost boosting algorithm to implement an ensemble learning
approach for solving a (binary) classification problem. 

The (one-dimensional) training data set is given in
Table 4.12 on page 352 of the textbook. The base classifier is a simple, one-level decision tree (decision
stump) (as explained on p. 303 of the textbook).

Determine the number of boosting rounds and show the result of each round (the weight distribution
wi’s at each round, which records are chosen at each round, the model (tree) obtained at each round,
the ϵ and the α at each round), as well as the result obtained with the final ensemble classifier.

What is the result of running your ensemble classifier on the following test data?
X = 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0

Submit source code (do not use a package) and your output (the round-wise results and the result on
the test data) following the styles of Figures 4.46, 4.49, and 4.50 of the textbook

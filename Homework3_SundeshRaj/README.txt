============================SXR3297 README TXT file for Homework3============================

Name : Sundesh Raj
Course: CSE-5334 (Data Mining)
UTAID : 1001633297

The zip file "Howework3_SundeshRaj.zip" consists of 2 sub folders, "q1" and "q2", along with the homework report
and the README file
=============================================================================================

NOTE: All codes need to be run in Spyder IDE with python 3.7 and above

q1/

- This folder contains solutions for Problem 1 question 1 and 2
- It contains 2 python files logisticRegression.py and problem1.py
- To run the file make sure both the python files are in the same folder and open it in Spyder IDE
- Run problem1.py on Spyder IDE to get the output for problem 1
- Run as is in Spyder IDE. No user input required

q2/

- This folder contains solutions for Problem 2 questions 1, 2, 3, 4, 5, 6 and 7
- This folder contains all the necessary python files along with saved ML models in the ./model and ./model_weight_image
  folders
- The output graphs are in the ./output folder
- For problems 1 through 5 we need to run the main.py file by changing the configurations and parameters in the
  parameter_setting.py file
  The main.py file picks up these arguments at runtime and uses them to perform the training and testing on the
  datasets
- We change the parameters like from using Cifar10 dataset to using Fashion_mnist dataset, and changing the type of ANN model
  needed for the problem
- For question 1 we keep the cifar10 dataset and use the CNN model to train and get the test predictions.
  Then the accuracy vs epochs and the loss vs epochs are saved in the ./output folder and also screenshots attached in the
  report
- Follow the same for questions 2 through 5
- For question 6 we remove the hidden layers and modify the mlp_model.py file and run the training on the OneLayer neural model
  We generate the accuracy and the loss graphs
  Then we run the weightsAsImage.py file to generate image based on the predicted values, first these values are converted to 
  vectors and then plotted to view them as images. These images are also attached as screenshots in the report document
- For question 7 first we run the plotROC.py file to generate the ROC plots for both the CNN model predictions as well as the
  Logistic regression prediction models.
  Then we run the twoClassCNNvsLR.py file to show how the accuracy and loss varies with respect to the training and test dataset.
  These graphs are also attached as screenshots in the report document.
  
Note: The report document also contains all the external resources and citations used for this project
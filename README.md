# CreditCardDefault_MachineLearning

## Objective

Assuming the role of a consultant, the project consisted on developing a model to predict which customers are likely to default on their credit card in the following month. The bank wants to use the model the decrease operational losses.

The Project objective consists of developing data mining models to predict which credit cardholders from Taiwan banks will likely default on the following month, by implementing a CRISP-DM methodology. The developed models seek to forecast which customers will likely default, or not, in the following month. 

By having this information, the financial institutions could adjust their procedures when assessing if a customer will likely default or not, when borrowing funds from the bank in the form of a credit card. Besides the client’s personal data it includes historical financial information.

**Business Understanding** phase of the CRIPS-DM framework is somehow compromised since we can’t contact the Taiwan banks and directly observe this credit card process. It would also be interesting to interact with their teams and assess if there’s additional information relevant to include in the models. Given the business objective, the target variable defined in this problem is a binary classification problem (default - 1 or not default - 0).

## Dataset
The data presented in default_credit_card_clients has a total of 30 000 observations (in the statistics it appears 30 001 since it is considering the label of each variable), described along 23 different features. Since the variables aren’t yet in the correct type, it misses some statistical information which will later be presented.

## Data Understanding 
The previous summary statistics and data overview also concern the Data Understanding phase. It is crucial to deepen the data understanding and examine the dataset and hence it is further verified the presence/absence of missing values, followed by a graphical visualization. It is confirmed that the data doesn’t present missing values

The variables’ type was transformed into Numerical and Categorical. The 9 categorical attributes are: Gender, Education, Marital status, History of Past Payments (from April to September) Now it is possible to obtain more complete summary statistics for the numeric features.

Regarding the credit amount, the minimum value observed was 10,000 NT dollars and the maximum was 1,000,000 NT dollars. The credit cardholders’ age included clients with ages between 21 and 79 years. 

Concerning the minimum and maximum values for the variables that reflect the historical bill statements from April – X17 to September – X12, it is possible to infer that the data has negative values as the minimum amount, which may suggest that a client paid a superior value of the due amount. These variables present a significant standard deviation. 

It is also possible to observe that the average of these amounts increases every month from April – X17 to September – X12, meaning that a customer bill statement increases, on average, every month. Lastly, the remaining numerical features reflect the amount of previous payments, again from April – X23 to September – X18. 

The minimum values observed are 0, meaning no payment was made. The average and 3rd quartile values are relatively small in comparison with the maximum values, suggesting a negative skew. Additionally, payments made in July, August and September are superior to the ones realized in the other months, which could possibly reflect the amounts that the clients spent during vacations.

### Categorical Variables
By examining the levels from the categorical features it is possible to infer that in Education (X3) there are categories that aren’t associated with any type of Education, which could be a typo when assessing the clients’ education. Also, when observing the marital status there are also 54 observation that don’t represent any of the available classifications. 

Additionally, in terms of historical past payments, the majority of the observations don’t have payments due or in delay. It is also possible to see that the number of months in delay increase from April – X11, to September – X6, meaning that if a payment is due in April, it is probable that it will be continued due in the following months. Finally, by analyzing the number of defaults (Y = 1), it is possible to conclude that the dataset is unbalanced, with only 22.12% of observations having default. Nonetheless, given the context of the problem, this number can be considered relatively high.

As one can observe, the majority of the credit cardholders are married or single. The default rate is slightly higher on the married ones (23.47%) than in single clients (20.92%).

Also, it is represented the defaults (or not) across the age range. It is possible to observe that clients between 25 and 31 have more occurrences of default. Nonetheless, in relative terms, the elder people have higher percentage rates, although there’s not enough data in those age ranges.

## Logistic Regression
It is a supervised learning method used to predict the probability of a binary ([yes/no], [0/1]) event. In this example, it was used to solve a classification problem. It can be said that Logistic regression models are models of the log of the odds ratio. In the example, it was proceeded to compute three times the logistic regression, the first with the data prepared, the second with oversampling to remove the imbalance and the third with under sampling to try to remove the imbalance too.

In this first result, it was obtained a relatively good accuracy of 0,82 in the train and 0,81 in the test sample. The precision, recall and, as a result, the F1 Score were under the expected as described in the table.

### Modeling and evaluating with oversampling to remove imbalance
Our second approach (oversampling) did not perform better, especially in Accuracy and Precision. Although got a slightly better result in the Recall as well in the F1 score and AUC.

### Modeling and evaluating with undersampling to remove imbalance
Our Third approach (undersampling) got almost the same result when compared to the second method (oversampling) - the obtained performance metrics were very similar, although not sufficiently enough to perform a good prediction for the data. As an initial conclusion we could infer that the Logistic regression did not fit well to predict our target in our data.

## Support Vector Machine
SVM is a supervised Machine Learning Algorithm that is widely used both for regression and classification models. In a simple way, the SVM algorithm consists on plotting each item as a point in a space with the same number of dimensions as features. The value of each feature is a certain coordinate. The classification is made by finding a boundary that differentiates the two classes clearly.

As one can observe from the table above, the model did not achieve very good results. Not only the obtained measures are very different between the train and test sets, but also the results themselves are very low. Precision is the highest performance measure obtained using the train set, although, when faced to the unseen observation of the test set, the model was not able to be consistent and the precision decreased severely.

Using the ROC curve (and AUC), it is possible to visually reach the same conclusions already explored above: the AUC is relatively low, despite being higher than in other models. This means that the model is predicting 0s as 1s and 1s as 0s.

Finally, the precision-recall curve was computed, in which it is shown the tradeoff between precision and recall for different thresholds. The desirable result is to obtain a high area under the curve thus meaning the model obtained both high recall and high precision, in other words, a low false positive rate and a low false negative rate. The curve obtained using the SVM algorithm reinforces the fact that both measures obtained are relatively low.

## K-Nearest Neighbours
The K-Nearest Neighbours assumes that similar things exist near each other. This model is considered one of the easiest to implement and can be very useful in classification problems, since it can easily identify the class of a particular data point. When selecting the number of K neighbours and the distance between data points is calculated, by assuming the similarity between a new data point and already available data, the KNN model is able to select the class most similar within the available ones.

This model does not learn using the training data, simply because the algorithm stores the datasets and performs the classification task when new data is available.

Looking at the confusion matrix, one can see that out of the training dataset , 77,88 % of clients did not default and 22,12% defaulted the next month. Of the percentage of clients that did not default, this model correctly identified 53,32%. However, out of the percentage of clients that defaulted, the model correctly identified only 13,47%.

Precision answers the question of what proportion of positive identifications was actually correct. Our model has a prediction of 0,354, meaning that when it predicts if a client will default in the following month, it is correct 35,4% of the time. Recall, on the other hand, tells us what proportion of actual positives were identified correctly: with a recall result of 0,61, the models identifies 61% of the clients that defaulted the next month.

The accuracy showed poor results from the train to the test dataset, with a possible explanation being the number of chosen neighbours. When the model is only selecting values closer to the data sample it is creating a more complex decision boundary for classification, therefore having more difficulty in presenting a correct generalization on the test dataset.

Finding the optimal number of neighbours is very important for this model, since it depends on it to correctly classify the test data set.

### Assessing the number of Neighbours
Approach based on the this [website](https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/#evaluatingthealgorithm).

Since when using this model, the value of k neighbours that will yield the best results for the dataset is unknown, a graph for the mean error rate of the data set (predicted values of test set) for each value of K (between 1 and 15) was plotted. The values with the smaller mean error should be the ones to consider and test how it impacts the accuracy, precision and recall of the test dataset.

## Neural Networks
Neural Networks is a useful model in recognizing underlying patterns between input and output variables, especially nonlinear and complex relationships, using clustering and classification to learn and improve. A simple NN can be described as a set of layers composed by individual neurons, each one with inputs, an activation function (one of the most used is the sigmoid function, that transforms neuron to a value between 0 and 1), and an output. In the majority of models, NN can be very complex and have many hidden layers.

Experimentation and changing parameters are crucial in neural networks, such as the number of neurons in each hidden layer, stopping criteria, learning rate etc, since the main challenge in the training is making sure you can get to a minimum error solution. Only by experimenting, you can find better solutions and make sure you have not reached a local minimum instead of a global minimum, as desired.

This model achieved an accuracy of 75,8%, although a high accuracy value does not necessarily indicate the model is good. When interpreting the confusion matrix, one can see that of the percentage of clients that defaulted, the model identified 13,09%, where for the clients that did not default, it identifies 62,72%.

When this model predicts that a client will default, it presents a correct result 46,34% of the time, although it only identifies 59,19% of the clients that will actually default in the following month. The accuracy increased from train to the test data and the precision dropped considerably. Both Recall, AUC and F1 Score got a slightly smaller results in the test data set.

## Naive Bayes
It is a classification algorithm for binary (two-class) and multi-class classification problems. The name “Naïve” assumes inputs as independent given the target value. That assumption performs well on data where this assumption does not hold. It works by calculating the probability of each class in the train dataset, then calculate the distance to neighbors and calculate the respective probabilities. The interpretation is based as a percentage of an input, as much higher the target level distribution, this means that feature is important.

In our model, the Naïve Bayes technique did not get good results on predicting our target. The accuracy on the test set was only 0,3193, the precision 0,2379, F1 score of 0,3799 and AUC of 0,5425. We could make an emphasis on the Recall that got a result of 0,9427 on the test set.

## Decision Tree
As in the regression assignment, the Decision Tree Model was also implemented. Since this is a classification problem the outcome of the DT will be, in this case, the mode response of the observations falling in the region of the tree, instead of the mean.

Firstly, the parameters implemented were only the maximum depth (4) and the random state.
By increasing the training instances, the gap between the Cross Validation Score and the Training Score decreases. Also from the 25,000 to 30,000 instances the scores are nearly the same although with a slightly increase.

The repayment status features in September (X6) have the higher coefficient Gini importance to the Model, meaning that the most recent repayment status available have the highest impact when predicting if a customer will default or not, in the following month.

In terms of accuracy, this model achieves significant results, both in the training and in the test set, being one of the top accurate models between the ones implemented. Although achieving precise results in the training set (93,7%), when faced to the test set, this measure reduces significantly, which can be a sign of overfitting. 

The same was verified in Recall and F1 Score values. Since the Recall is relatively low and inferior to Precision, it indicates a significant presence of False Negatives, meaning that the model classified a considerable number of clients as not defaulting in the following month, when in reality, the data states otherwise.

Overall and considering the AUC and the ROC curve to compare the performance of Decision Tree Model with the other implemented approaches, it is possible to conclude that the Decision Tree produced a somewhat good score in terms of AUC in the training set, however a poor result in the test set.

### Hyper tunning
By hyper tunning some Decision Tree parameters it was possible to achieve some better and worst results in comparison with the first Decision Tree. In terms of accuracy, it is observed a higher value in the training set, although a worse performance in the test set. In terms of classifying the defaults among the actual and false defaults the hyper tunned Decision Tree also delivered a worse score. 

On the other hand, Recall was significantly better in both training and test set, meaning that this model captured more false negative classifications. Lastly, the F1-Score and AUC was also slightly better. Overall, if one only compared the AUC, the hyper tunned Decision Tree delivered better results, although less accurate and precise when faced with new observations, which may indicate signals of overfitting.

## Random Forest
The Random Forest is an ensemble model. It is associated with the bagging algorithm with the following modification: instead of the following decision tree being created with the previous bootstrapped sample, which considers all features as candidate variables, it only selects a random number of variables. The Random Forest algorithm from the sklearn ensemble models was parametrized with 100 trees (in the forest), with the maximum of depth being 3. It was also assigned a random state.

By observing the model results it is possible to infer that it produced relatively accurate results both in the training and in the test set, meaning that overall produced approximately ¾ of corrected predictions. 

In terms of precision, there’s an evident discrepancy between the results of the training and the test set. This means that in the training set, the Random Forest Model captured a reasonable number of credit card clients’ default, among the ones who were classified as defaults. In the test set, this situation wasn’t verified which highlights the presence of False Positives classification (Type I Error). 

The same situation was observed in the Recall evaluation method – big discrepancy between the train and test set results. The 56% obtained in the test set emphasizes the presence of false negatives (Type II Error). Since the F1-Score can be viewed as the harmonic mean of precision and recall, the same behavior was observed between the train and test set.

Considering the results for the AUC, which is the area under the ROC curve, this model achieved a value of nearly 80% in the training set, with a decrease of approximately 10 percentage points in the training set. As stated previously, there’s evidence of a higher presence of false positives and false negatives in the test set, which translates into a lower AUC in the test set.

In order to obtain some interpretability, it was implemented the feature importance algorithm (also known as Gini importance) from the scikit-learn package. By applying this algorithm it is possible to observe that the most relevant features are the ones related with repayment status being on time in September, August, July and June being, with the outstanding variables representing very low or non-existing importance.

### Hyper tunning
We'll use Randomized Search to find the best parameters to implement in the Random Forest Model Followed some instruction in this [website](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74).

In order to optimize the Random Forest Model it was executed the Randomized Search CV from the Scikit Learn package to obtain the best parameters of the model automatically, instead of trial and error. This approach selects a grid of parameters’ ranges and performs the K-Fold CV with each combination of values.

When comparing the initial Random Forest with the one with the hyper parameters, it is possible to observe a considerable increase in the training set, with results almost above 95%. In terms of Accuracy and Precision, it is verified a significant increase also in the test set, although the difference to the training set remains high. In terms of Recall, when faced with new data, the model classified more instances that have defaulted as no defaults (False Negatives). 

Overall, although the hyper tunned model delivered better results in the training set, the test set measures remained low, with the exception of Accuracy. This is evidence of overfitting the data. Additionally, if we compare both AUC from the test sets, the optimal parameters model presented worst results.

## X Gradient Boosting
XGB is an ensamble machine learning algorithm based on the decision tree algorithm. These decision tree algorithms are considered to be one of the best when dealing with structured data. XGB is a more powerfull gradient boost where the errors are minimized by gradient descent algorithm.

The measures obtained using the XGBoost algorithm are somewhat better than the makority of other models. The precision, accuracy and AUC obtained with the train set are not bad although, as it is easy to observe, there results are much different from the ones obtained using the test set. This clearly shows us that oversampling is present.

Using the ROC curve (and AUC), it is possible to visually reach the same conclusions already explored above: the AUC is higher than the one obtained from other models. Although, when faced with new observations (test set), the model decreased its performance.

## Shapley
By using the Shapley values, one is trying to interpret the model through the understanding of the individual features’ contribution to the output and their interactions.

By visualizing the summary plots, the variable with the highest importance is the repayment status in September with 2 months delay (which is also visible in other models). The variable with the highest dispersion, meaning with higher extreme values on the x-axis, is the feature X14: The amount of bill statement in July. This might be related to the initial assessment of the data, which relates the amount of bill statement in July and the holidays’ spending. Hence, a person with a higher or lower bill statement in July, will have a significant impact on the model’s prediction.

Overall, this instance contributes negatively to the default classification. It is possible to observe the feature’s impact on the prediction (-0.07). In red it is represented the features that contribute negatively to default and in blue the positive contribution. As it has been seen, the X6_2 (Repayment status in September being 2 month delay) is the feature with the highest importance, and in this example, has a value of 0. So, makes sense the rationale of this instance contributing negatively to the classification of default.

## Conclusions
There can be two different approaches in order to choose the preferred model. The first would be to focus on general performance measures, such as the F1-Score and the AUC. Here, the XGB model was the one that retrieved higher values when compared to the other models: an AUC of 70,37% in the test set (with a difference of 0,016 to the training set) and a F1-Score of 52,34% (when compared to 68,49% obtained with the training set). 

It is important that the difference between the obtained measures when using the two sets are as close as possible. Furthermore, taking into consideration the problem the model is trying to solve (predict which customers are going to default on the following month), the precision measure is the one that should be prioritized, as one wants to predict as many default observations as possible, even the false positives. 

Considering this, the model that retrieved the highest precision was the Decision Tree (65,80%). Although, when compared to the training set, this represents a difference of 0,28 and, despite the Logistic Regression model had obtained a slightly lower value for this performance measure (65%), the difference to the train set is only 0,03 and thus, should be the chosen model when one wants to prevail precision as the most important measure.

**Hope you find this project interesting**

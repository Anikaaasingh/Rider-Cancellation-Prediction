# Rider-Cancellation-Prediction
It consisted of an unbalanced dataset of 450,000 rows with 97:3 being the ratio of non-cancelled data to cancelled data of delivery orders.
Most significant improvement was brought by the Class Weights method as it penalised more if got the minority data prediction wrong. 
Along with that, effective EDA alongwith some data manipulation led us to AUC-ROC score of 0.81 and 0.83 on public and private leaderboards respectively.

# Objectives
The key objective of this report is to analyze the given dataset and generate insights 
to figure out correlations between variables in the dataset. The analysis will be largely 
focused on order cancellations and factors associated with it. Our analysis is 
generalized over orders and riders, making it applicable to any new order or rider 
outside of dataset.
 The key questions that this report will answer are: 
- What properties of orders and riders indicate cancellation and upto what 
extent?
-How are variables that affect cancellation related with each other?

 # Feature Engineering
We created some new_features to enhance the analysis:
-“allot_order_delta”: difference between “allot_time” and “order_time”.
-“accept_allot_delta”: difference between “accept_time” and “allot_time”.
-“pickup_accept_delta”: difference between “pickup_time” and “accept_time”.
-“total_distance”: sum of first mile and last mile distances.
-“delivered_allot_ratio”: ratio of “delivered_orders” and “alloted_orders”.

# Methodolgy
-As the analysis is focused on cancellations, it makes sense to study how many cancellations are made subject to certain conditions and compare them with how 
many deliveries are made under same conditions. We define a variable named “cancellation_rate” defined as the probability that a randomly sampled order from the 
dataset under some condition S on variable X was cancelled :
Cancellation Rate=P(cancelled|X(s)
-We use this methodology throughout the report to analyze how the variables in dataset affect cancellation. It is very important to note that cancellation rate is very sensitive to the denominator, say there is a certain condition on a variable that is satisfied by only one sample in the dataset and if that one sample turns out to be a cancelled order then cancellation rate is simply 100% which is absurd. So for an inference based on cancellation rate to be statistically significant, total number of samples satisfying the given condition must be sufficiently large, we chose this minimum limit as 100. The report only contains statistically significant inferences.

# Rider Analysis:
In this we analyzed the rider specific variables and their relation with cancellation. We have a figure below for a quick overview of rider related variables 
and their distributions.
![Screenshot (343)](https://github.com/Anikaaasingh/Rider-Cancellation-Prediction/assets/96921017/cb20edfc-5ac7-4caa-8625-f086764a4944)

# Conclusion:
The probability of cancellation of an order beforehand can be estimated to a reasonably good accuracy considering the relationship between given variables. We trained a stacked classifier containing XGBClassifier, RandomForestClassifier, DecisionTreeClassifier, AdaboostClassifier, GaussianNB and finally a Logistic Regression as final estimator. Our model was able to achieve an accuracy of 83% on test data. In the bar graph below we have plotted the feature importances of our model computed using permutation 
importance method.
![Screenshot (344)](https://github.com/Anikaaasingh/Rider-Cancellation-Prediction/assets/96921017/0ad2808b-fe91-4270-8546-cca038f0e72f)

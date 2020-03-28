# Grocery recommender system

#### About this project

This project builds a recommender system for a grocery store.  It utilizes data on customer behavior over a 52 week period in order to predict items a customer is likely to purchase in the following week (illustrated below):

![](Images/Timeline.PNG)

A recommender engine such as this could be utilized to:

1.	Create a personalized digital flyer highlighting to each customer the items on sale that week that are most relevant for them
2.	Generating automated weekly shopping lists
3.	Reminding customers shopping on-line of items they may have forgotten to purchase i.e. items with high predicted relevancy for that week they did not purchase

The recommender system is built utilizing matrix factorization techniques commonly used in recommender systems but will also be combined with ML classification models that will add contextual features e.g. average customer purchase cycle for an item in order to predict the probability that a customer will purchase an item in a given week.  


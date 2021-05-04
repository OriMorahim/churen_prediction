INTRO:
This challange practice at predict the possibillity of a user to churn.
Churn defiend as a user which the total worth of equity he posses in the end of the flowing month is less than 25$.


ASSUMPTIONS:
1. In real time the scoring popultaion includes only useres which currently have more than 25$ worth of equity 


PREPROCSS:
** If I had more time I would:
1. Clean the data better (I nothiced there is issue with the dates - there are dates that yet to be come)
2. Find a better solution to NA values


FEATURE ENGINEERING:
The feature engineering will be pratice at:
1. lag features
2. moving averages
3. time differences

Note, all the feature engineering process can be done to the train and test together, as there are not features whice might cause a leakage 

** If I had more time I would:
1. Add features regarding the user investment type
2. Add the category features
3. Add difference features between lags

TRAINING:
1. it seems like there is a slighly imbalanced situation, still nothing dramatic
therefore i'll chose a model that can handle imbalanced situations (GBM), without use a more active way such as SMOTE, sunder/over sampling.

2. The division to train, test, validation will be based on a time to avoid the look-ahead bias. Meaning, a user future data wont be found in the train. 

** If I had more time I would:
1. add cross validaiotn to set hyper param 
2. Try to segment the users to cluster and generae a model to each cluster  
# PROJECT 2 - AMES HOUSING DATA AND KAGGLE CHALLENGE

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Problem-Statement" data-toc-modified-id="Problem-Statement-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Problem Statement</a></span></li><li><span><a href="#Executive-Summary" data-toc-modified-id="Executive-Summary-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Executive Summary</a></span></li><li><span><a href="#Data-importing-and-cleaning" data-toc-modified-id="Data-importing-and-cleaning-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Data importing and cleaning</a></span></li><li><span><a href="#Data-Dictionary" data-toc-modified-id="Data-Dictionary-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Data Dictionary</a></span></li><li><span><a href="#Exploratory-Data-Analysis" data-toc-modified-id="Exploratory-Data-Analysis-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Exploratory Data Analysis</a></span><ul class="toc-item"><li><span><a href="#Exploring-distribution-of-SalePrice---First-Look" data-toc-modified-id="Exploring-distribution-of-SalePrice---First-Look-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Exploring distribution of SalePrice - First Look</a></span></li><li><span><a href="#Categorical-features-EDA" data-toc-modified-id="Categorical-features-EDA-5.2"><span class="toc-item-num">5.2&nbsp;&nbsp;</span>Categorical features EDA</a></span></li><li><span><a href="#Numerical-features-EDA" data-toc-modified-id="Numerical-features-EDA-5.3"><span class="toc-item-num">5.3&nbsp;&nbsp;</span>Numerical features EDA</a></span></li><li><span><a href="#Continuous-features-shortlisting" data-toc-modified-id="Continuous-features-shortlisting-5.4"><span class="toc-item-num">5.4&nbsp;&nbsp;</span>Continuous features shortlisting</a></span></li><li><span><a href="#Correlation-of-Numeric-features" data-toc-modified-id="Correlation-of-Numeric-features-5.5"><span class="toc-item-num">5.5&nbsp;&nbsp;</span>Correlation of Numeric features</a></span></li></ul></li><li><span><a href="#One-Hot-encoding-and-Feature-Engineering" data-toc-modified-id="One-Hot-encoding-and-Feature-Engineering-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>One-Hot encoding and Feature Engineering</a></span></li><li><span><a href="#Baseline-model-score" data-toc-modified-id="Baseline-model-score-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Baseline model score</a></span></li><li><span><a href="#Model-Prep-Train-Test-Split-and-StandardScaler" data-toc-modified-id="Model-Prep-Train-Test-Split-and-StandardScaler-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Model Prep Train-Test-Split and StandardScaler</a></span></li><li><span><a href="#Model-evaluation-using-cross-validation" data-toc-modified-id="Model-evaluation-using-cross-validation-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Model evaluation using cross validation</a></span></li><li><span><a href="#Model-Fitting,-Evaluation-and-Tuning" data-toc-modified-id="Model-Fitting,-Evaluation-and-Tuning-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Model Fitting, Evaluation and Tuning</a></span></li><li><span><a href="#Re-run-Lasso-with-reduced-features" data-toc-modified-id="Re-run-Lasso-with-reduced-features-11"><span class="toc-item-num">11&nbsp;&nbsp;</span>Re-run Lasso with reduced features</a></span></li><li><span><a href="#Generating-predictions-for-Kaggle's-Test-data" data-toc-modified-id="Generating-predictions-for-Kaggle's-Test-data-12"><span class="toc-item-num">12&nbsp;&nbsp;</span>Generating predictions for Kaggle's Test data</a></span></li><li><span><a href="#Kaggle-Submission-Result" data-toc-modified-id="Kaggle-Submission-Result-13"><span class="toc-item-num">13&nbsp;&nbsp;</span>Kaggle Submission Result</a></span></li><li><span><a href="#Conclusion-and-recommendations" data-toc-modified-id="Conclusion-and-recommendations-14"><span class="toc-item-num">14&nbsp;&nbsp;</span>Conclusion and recommendations</a></span></li></ul></div>

## Problem Statement

The Ames Housing Dataset is an exceptionally detail and robust dataset with over 80 columns of different features relating to houses. In this challenge, we are expected to use the data to create a regression model that predicts the price of houses in Ames, IA. We are free to use any and/or all features that are present in this dataset.

Various stakeholders (Buyers, Sellers, Realtors, Tax agencies) normally measure or predict housing prices in terms of their intuition or the factors of price known to them like location and size. However, **there are as many as hundreds of factors affecting housing prices, which makes it very difficult to assess and quantify the relationship between factors and prices.**

Thru the power of data analytics and machine learning. This project involved training several machine learning models (Linear, Ridge, Lasso and ElasticNet regression) that use the house features and attributes to predict the sale price of houses in Ames, Iowa. **We will examined the features to determine which features are important and which are not, developed multiple machine learning models to find the best model to predict sale prices.**

**The overall success of the prediction model will be based up its Root Mean Squared Error (RMSE).**


## Executive Summary

We begin the project with Kaggle's data file that consists of 2051 rows by 81 columns and a data dictionary to aid our understanding of the data. By studying the data dictionary, we gain insights into the data types, their meaning and the range of values that helped us in making decisions over how to deal with missing values, outliers and even intuitively dropping some features that are irrelevant to the objective.

After performing EDA, we were able to apply better decisions in the treatment of Zero values in the data and was able to shortlist 27 relevant base features for further processing. We used simple feature engineering based on logical assumptions with limited domain knowledge and when stacked with those generated by hot-encoding, we were again faced with 127 features. Using correlation analysis, we trimmed that down to 30 most significant features.

We passed that data through a baseline linear regression model to established a baseline score. Then attempted to regularize the other various models (Linear, Ridge, Lasso, ElasticNet) using model tuning to ensure that we had chosen the best model that could make the best predictions on unseen data.

Using the production model (Lasso), we were able to reduce the measuring metric, RMSE that greatly improved on the performance of predictions as compared to the baseline linear regression model. Another 7 features were also eliminated by this model. This improvement was eventually validated by a very much improved Kaggle submission score as compared to the first submission.

The final set of 23 features deemed to be the best predictors were:

`OverallQual` `ExterQual` `KitchenQual` `GarageCars` `GrLivArea` `BsmtQual` `GarageArea` `1stFlrSF` `GarageFinish` `FireplaceQu` `TotRmsAbvGrd` `HeatingQC` `Neighborhood_NridgHt` `MSSubClass_60` `Foundation_CBlock` `BsmtFinType1` `Exterior1st_VinylSd` `MasVnrType_Stone` `LotShape` `MSZoning_RM` `Neighborhood_NoRidge` `MasVnrType_BrkFace` `Neighborhood_StoneBr`


## Data importing and cleaning

In this step, we took the systematic approach of:

- identifying and treating the issues identified in the analysis of data dictionary
- fixing columns naming differences between data file and data dictionary
- identifying obvious features to remove due to excessive missing values
- fixing Null (NaN) values in some features according to their description in data dictionary
- imputing actual ordered values for the categorical ordinal features so that they will be treated correctly mathematically
- fixing data type differences (if any)
    
We felt that getting the data in an organized form initially was crucial to proper Exploratory Discovery Analysis (EDA) step that follows. We started the process with 2051 rows by 81 columns, ended with 2051 rows by 75 columns.

## Data Dictionary

The data was compiled by Dean De Cock and published in Kaggle. Evaluating the data dictionary is an important first step in any data science project. From the dictionary we noted that Tab character was used to separate variables in the data file and there were 82 columns which included 23 nominal, 23 ordinal, 14 discrete, and 20 continuous variables (and 2 additional observation identifiers).

- Categorical = 23 nominal + 23 ordinal + 1 additional observation identifier = 47
- Numerical = 14 discrete + 20 continuous + 1 additional observation identifier = 35
- We will follow the categorical and numerical features based on the data dictionary to ensure consistency.

We also found that:
1. Data dictionary specifically mentioned 5 unusual records (true outliers) whereby they were very large houses priced relatively appropriately. It was recommended by Prof Dean to remove any houses with GrLivArea more than 4000 square feet which will eliminate these 5 unusual observations. We will take note and remove them as we process the data.


2. Also, the following columns were named differently between the data file and data dictionary. Since we did not want to alter the data file, we will adjust the compiled data dictionary list to follow the data file.

    - _Data Dict = "Exterior 1", "Exterior 2"_ ; _Data file = "Exterior 1st", Exterior 2nd"_
    - _Data Dict = "Sale Condition"_ ; _Data file = no such column_
    - _Date Dict = "Bedroom", "Kitchen", "3-Ssn Porch"_ ; _Data file = "Bedroom AbvGr", "Kitchen AbvGr", "3Ssn Porch"_


3. From reading the description of each feature, we noticed that `Id` and `PID` was unlikely to offer any insights so we will remove these as well.


4. There were some close relationship between features, whereby if one was being eliminated or showing a zero value, it will not make sense to the others. For example:
- TotalBsmtSF refers to 'Total square feet of basement area', so if this is zero it is likely to mean that there is no basement. Consequentially, there should not be any values for the related features of BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath, BsmtHalfBath. In reverse, if any of the other features had values and TotalBsmtSF is zero. This implied that we need to impute TotalBsmtSF with some value, likely to be the mean. Another more obvious example will be Fireplaces and FirePlaceQu, where their relationship should be either True/Ture or False/False.
- However, due to the complexity of cross validating every single relationship and the limited domain knowledge. We will not be actively investigating this aspect but will treat those that standout during the EDA process.


5. Certain features should be considered collectively, for example RoofStyle and RoofMatl, MasVnrType and MasVnrArea. Features like these could be targets for feature engineering in later steps if they are selected.

## Exploratory Data Analysis
    
In this step, we went deeper to analyze the features:

- look at basic statistic and distribution of dependent target, SalePrice
- missing values and imputing them appropriately
- identifying and addressing outliers
- clean up other issues discovered post data cleaning step

This EDA step would give us a first cut selection of features that are most likely to have an impact on SalePrice.

### Exploring distribution of SalePrice - First Look



![pic2.png](attachment:pic2.png)

Initial data cleaning showed that a number of features had missing values, which would need to be managed appropriately before modelling.  Before doing this we examined the target variable’s (i.e. the SalesPrice) descriptive statistics.

A mean sale price of 181,470 and median of 162,500 on the histogram indicated positive skew. Sale prices range from 12,789 to 611,657 with the 95th percentile in the boxplot that was considerably lesser than the maximum sale price, indicating a numbers of outliers need to be treated.

### Categorical features EDA

In this step, we choose to make plots of histogram and boxplot against saleprice to assist in the EDA.

In each of the frequency plots, we added a horizontal line which was the 50% count of the feature to assist in visualizing any features that exhibit a strong bias towards certain selections. These features are considered to be contributing to the skew in SalePrice distribution and should be removed from consideration.

In the box plots, we were able to visualized the extremity of outliers in the features to consider for removal or treatment on those extremities.

![pic3.png](attachment:pic3.png)

For example, `MsSubClass` showed a lot of outliers at both max and min ends in the boxplot. Although this feature was shortlisted we had to keep in mind that it will require further analysis to treat those outliers appropriately.

On the other hand, `Condition2` was a straight forward decision to drop this feature as there was an extreme bias (i.e. around 50% concentration) towards a sub-category that was likely to skew the saleprice distribution.

At the end of this EDA on categorical features, we ended up:

`Street`, `LandContour`, `LotConfig`, `Condition1`, `Condition2`, `SaleType`, `RoofStyle`, `Heating`, `Utilities`, `LandSlope`, `ExterCond`, `BsmtCond`, `BsmtExposure`, `BsmtFinType2`, `Electrical`, `Functional`, `GarageQual`, `GarageCond`, `PavedDrive`, `BldgType`, `RoofMatl`, `CentralAir`

**22 features excluded**

We included the following because they have more distinct categories and did not appear to had a strong bias (i.e. <50% concentration). We also noted the outliers in these that we may have to treat them in the next steps.

`MSSubClass`, `MSZoning`, `Neighborhood`, `HouseStyle`,`Exterior1st`, `Exterior2nd`, `MasVnrType`, `GarageType`, `Foundation`, `LotShape`, `OverallQual`, `OverallCond`, `ExterQual`, `BsmtQual`,  `BsmtFinType1`,  `HeatingQC`, `KitchenQual`, `FireplaceQu`,  `GarageFinish`

**Shortlisted - 19 features**

### Numerical features EDA

In this step, we choose to make plots of histogram and boxplot against saleprice to assist in the EDA.

In each of the frequency plots, we added a horizontal line which was the 50% count of the feature to assist in visualizing any features that exhibit a strong bias towards certain selections. These features are considered to be contributing to the skew in SalePrice distribution and should be removed from consideration.

In the regplots, we were able to visualized the extremity of outliers in the features to consider for removal or treatment on those extremities.

![pic4-2.png](attachment:pic4-2.png)

For example, we observed that `YearBuilt`, `YearRemod/Add` and `GarageYrBlt` had showed a clear positive linear relationship to saleprice. In this process, we identified features that need to be treated and get it done.

    - YearRemod/Add had a concentration of values in the year 1950 that may need to be treated.    
    - GarageYrBlt had an irregular outlier in the future year and since it is only 1 row, we will drop the row.
    
Another example, `GarageCars` `KitchenAbvGr` had positive and negative linear relationship respectively to saleprice as the discrete value increased. Intuitively, in the real world, as the value of these features increase. It was also likely that the sale prices will trend upwards as well. (e.g. more Bedrooms or more car capacity in the garage usually fetch a higher price). Upon reading the data dictionary, we found these to be logical discrete values. Hence we kept them in the selection list.

`MoSold` had almost a flat relationship to saleprice. This was another obvious feature to drop because it did not demonstrate any impact to saleprice.

Features that demonstrated some form of positive/negative linear relationship to saleprice were shortlisted. They were:

`YearBuilt` `YearRemod/Add` `BsmtFullBath` `BsmtHalfBath` `FullBath` `HalfBath` `BedroomAbvGr` `KitchenAbvGr` `TotRmsAbvGrd` `Fireplaces` `GarageYrBlt` `GarageCars` `MoSold`

**Shortlisted - 13 features**

### Continuous features shortlisting

Here we choose to make plots of Seaborn's regplot and stripplot against saleprice to assist in the EDA.

In the regplots, we can see the effect of the outliers and zero values as depicted by the diverging fan of the regression line. The bigger the red shaded area mean the bigger effect these outliers and zero values had on the saleprice variation. 

![pic5.png](attachment:pic5.png)

Next we used stripplot with additional 3 vertical lines to assist in determining the maximum threshold value to use for treating outliers. Although an overlay of swarmplot over a boxlot would be a better choice of graphic. It was not possible here due to having over 2000 data points that could not fit nicely into the constraint figure size. Thus, we calculated the threshold and plotted the overlay. Thresholds were calculated as follows:

    - IQR = 75% quartile - 25% quantile
    - Max threshold = 75% quantile + 1.5 * IQR (note: anything above this is an outlier and should be treated)
    - Min threshold = 25% quantile - 1.5 * IQR (note that this may not always max sense here as there was no zero saleprice
    - median = 50% quantile
    - mean = mean of population

![pic12.png](attachment:pic12.png)

There are 19 numerical continuous features (excluding `SalePrice`) and from analyzing the plots above, we observed two distinct points:

1. Some features had a linear relation with saleprice so we will select these. However, these features had a lot of outliers (as shown in the extent of the red shaded area in the regplot and number of data points above the max. threshold on the stripplot). We decided to set a threshold for each selected feature to identify the outliers, then drop the outliers. These included:

`LotFrontage` `LotArea` `BsmtFinSF1` `BsmtUnfSF` `TotalBsmtSF` `1stFlrSF` `GrLivArea` `GarageArea` `OpenPorchSF`

There are 3 border line features where the number of zeros is over or close to 50% of the population (i.e. 2050/2 = 1025). Upon consulting the data dictionary on their description, it seemed logical that these features may have zero values. For example, if there is no WoodDeck in the house, this value would be zero. Thus, we decided to leave these as zeros.

`MasVnrArea` `2ndFlrSF` `WoodDeckSF`

**12 shortlisted**


2. Some features had a large cluster (more than 50% of rows i.e. 2050/2=1025) of Zeros. If we impute these with the mean, it merely shifts the cluster to the mean. We can try to infer from their data dictionary description like we did before , but due to the excessive number of zeros, the median naturally become zero as well. Thus, we decided to drop these features from the selection. These included:

`BsmtFinSF2` `LowQualFinSF`  `EnclosedPorch` `3SsnPorch` `ScreenPorch` `PoolArea` `MiscVal`

**7 removed from selections**

These features were then treated according to the decision taken.

### Correlation of Numeric features

![pic6.png](attachment:pic6.png)

We plotted the Pearson correlation heatmap and saw the correlation of numerical independent features with the output variable SalePrice. We will only select features which has correlation of above 0.5 (taking absolute value) with the output variable.

By sorting the list, we could see that the top features with highest positive correlation coefficient (>0.5) were:

    GrLivArea        0.668579
    YearBuilt        0.631692
    GarageCars       0.626435
    GarageArea       0.601429
    GarageYrBlt      0.598366
    YearRemod/Add    0.589632
    TotalBsmtSF      0.588831
    1stFlrSF         0.554601
    FullBath         0.548719
    
These were then put plotted into any heatmap to eliminate features with high collinearity to each other.

![pic7.png](attachment:pic7.png)

Now we can visualize better the correlation between variables, one of the assumptions of linear regression is that the independent features need to be uncorrelated with each other. If these features are highly correlated (>80%) with each other we should keep only one of them.

- `GarageYrBlt` vs `YearBuilt` - 0.86 (dropping `GarageYrBuilt` as it had lower correlation with `SalePrice`)
- `TotalBsmtSF` vs `1stFlrSF` - 0.85 (dropping `TotalBsmtSF` as it had lower correlation with `SalePrice`)

## One-Hot encoding and Feature Engineering

After applying one-hot encoding to the selected list of 28 features, we ended up with 125 features.

This was followed by performing feature engineering on the remaining features, specifically the categorical nominals that were not taken care of by .get_dummies(). They are `YearBuilt` and `YearRemod/Add`.

1. `Year Remod/Add` (Discrete): Remodel date (same as construction date if no remodeling or additions)

We engineered `YearRemod/Add` into a categorical feature by replacing it with a new feature called `RemodHist`, of which the data value would be:

    - 1 if the value is not equal to YearBuilt (meaning a remod was done before)
    - 0 if the value is equal to YearBuilt (meaning no remod history).

This was based on the data dictionary description interpretation. We also added prefix fe_ to help in identifying engineered features.

2. `Year Built` (Discrete): Original construction date

We engineered `YearBuilt` into a continuous feature by replacing it with a new feature called `fe_Age`, which was the current year (2020) minus `YearBuilt`. Also added prefix fe_ to help in identifying engineered features.

With so many features (125) back in the picture, we need to trim this down so we took the absolute correlation values of the features against SalePrice and selected only those that were more than 0.25. This was to avoid excessive removal of features too early in the process. We were left with 30 features and we plotted the correlation heatmap again for the numeric features.

![pic8.png](attachment:pic8.png)

From the heatmap we observed that there was a good spread of original and hot-encoded features, as well as the correlation between features and sale price. Two features stood out for their high correlation to each other, namely `Exterior2nd_VinylSd` vs `Exterior1st_VinylSd` (0.98); and `MasVnrType_BrkFAce` vs `MasVnrType_None` (-0.82). Although logically this was a good time to drop these features due to their high collinearity. We noticed that these were features engineered by .get_dummies function and each pair belonged to the same feature category. We decided to leave them as it is for now since it was only 4 features.

## Baseline model score

Having finalised the features, we run a linear regression with cross validation to determine the baseline score.

|Model Type|Validation Method|R2|RMSE|
|:---|:---|:---|:---|
|Baseline (Unseen data)|LR cross validation|       0.856 | 	 22,146|

Baseline models are often expected to be the most liberal and lacking so in theory, whatever models we tuned and used for production, must be performing better than these baseline models.

Looking at 𝑅2, we could interpret this as 𝑅2 value of 0.86 means around 86% of the variability in sale price was explained by the x-features in our model. It is however, important to note that 𝑅2 is ONLY INTERPRETABLE WITH LINEAR REGRESSION.

It is hard to judge what would be a good 𝑅2 range at this point as the features may very well change as we proceed. 

**Thus, for Baseline comparison, we can expect a 𝑅2 score around 0.86 (tolerance +/- 5%) and RMSE score to be lower since the key measurement metric of this project was RMSE.**

## Model Prep Train-Test-Split and StandardScaler

We applied Train-Test-Split and StandardScaler to prepare the model data. Followed by instantiating the various regression models that we intend to evaluate.

![pic9-2.png](attachment:pic9-2.png)

## Model evaluation using cross validation

We used cross_val_score to evaluate our models.
- train data set = data set given by kaggle that was cleaned and processed in previous steps.
- train bloc = portion of the train data set use to train the model
- test bloc = portion of the train data set use to validate the model

This was the result:

|Model Type|Validation Method|R2|RMSE|R2 var +/- 5%|RMSE better?|Remarks|
|:---|:---|:---|:---|:---:|:---:|:---|
|Baseline (Unseen data)|LR cross validation|       0.8577| 	  22,146|---|---|---|
|---|---|---|---|---|---|---|
|Train bloc|LR cross validation|	  0.8565| 	 22,543|Yes|No|Lower bias, Increase variance|
|Train bloc|Ridge cross validation|	 0.8571| 	 22,497|Yes|No|Lower bias, Increase variance|
|Train bloc|Lasso cross validation|	 0.8566| 	 22,536|Yes|No|Lower bias, Increase variance|
|Train bloc|ElasticNet cross validation|	 0.8570| 	 22,507|Yes|No|Lower bias, Increase variance|

When we applied the basic models (without tuning) to the train bloc and comparing it against the baseline. We saw that all the models had lower bias (R2) but it is within the tolerance (+/- 5%) that we set. We also saw an increase in variance (RMSE) but not by a large margin. This was more or less expected as the number of rows had decreased by almost 1/3 of the total train data set with 2050 rows.

This was a typical case of overfitting where the models matches the training data "too closely”. Learning from the noise in the data, rather than just the signal. This happened because we are evaluating the models by testing it on the same data that was used to train it. Hence, creating a model that is "too complex".

The impact of overfitting was that the models will do well on the training data, but won't generalize to out-of-sample
data. The models will have low bias, but high variance which is exactly what we had observed here.

This observation brought out the need for Regularization. Regularization is a method for "constraining" or "regularizing" the size of the coefficients, thus "shrinking" them towards zero. It reduces model variance and thus minimizes overfitting.
If the model is too complex, it tends to reduce variance more than it increases bias, resulting in a model that is more likely to generalize. Our goal was to locate the optimum model complexity, and thus regularization is useful when we believe our model is too complex. We shall observed this in the next step.

Note: we added ElasticNet here as we intent to determine in next step if it will produce a better result.

## Model Fitting, Evaluation and Tuning

We then fitted the model to the train bloc, and evaluate the train and test bloc scores. This was the result:

|Model Type|Validation Method|R2|RMSE|R2 var +/- 5%|RMSE better?|Remarks|
|:---|:---|:---|:---|:---:|:---:|:---|
|Baseline (Unseen data)|LR cross validation|       0.8577| 	  22,146|---|---|---|
|---|---|---|---|---|---|---|
|Train bloc|LR cross validation|	  0.8565| 	 22,543|Yes|No|Lower bias, Increase variance|
|Train bloc|Ridge cross validation|	 0.8571| 	 22,497|Yes|No|Lower bias, Increase variance|
|Train bloc|Lasso cross validation|	 0.8566| 	 22,536|Yes|No|Lower bias, Increase variance|
|Train bloc|ElasticNet cross validation|	 0.8570| 	 22,507|Yes|No|Lower bias, Increase variance|
|---|---|---|---|---|---|---|
|LR on Train bloc|fit and score|	 0.8657| 	 21,871|Yes|Yes|Increase bias, Lower variance|
|Ridge on Train bloc|fit and score|	 0.8653| 	 21,903|Yes|Yes|Increase bias, Lower variance|
|Lasso on Train bloc|fit and score|	 0.8651| 	 21,924|Yes|Yes|Increase bias, Lower variance|
|ElasticNet on Train bloc|fit and score|	 0.8648| 	 21,944|Yes|Yes|Increase bias, Lower variance|
|---|---|---|---|---|---|---|
|LR on Test bloc|fit and score|	  0.8517| 	 20,783|Yes|Yes|Increase bias, Lower variance|
|Ridge on Test bloc|fit and score|	 0.8515| 	 20,796|Yes|Yes|Increase bias, Lower variance|
|Lasso on Test bloc| fit and score|	 0.8510| 	 20,833 |Yes|Yes|Increase bias, Lower variance|
|ElasticNet on Test bloc|fit and score|	 0.8517| 	 20,783|Yes|Yes|Increase bias, Lower variance|

**1. Train bloc Fit and Score vs Train bloc Cross Validation:**
- Although there was an increase in bias (R2), it was within tolerance.
- Variance had decreased considerably across models using default hyperparameters and optimal alphas. Among the models, Lasso Regression had the largest reduction in RMSE of -2.71% (21924/22536).

**2. Test bloc Fit and Score vs Train bloc Fit and Score**
- Consistent in increase in bias but within tolerance and decreased variance considerably across all models. Again, among the models, Lasso Regression registered the largest reduction in RMSE of -7.56% (20833/22536).

**3. Train and Test bloc vs Baseline Cross Validation**
- In comparison to baseline, all the models had lower RMSE than baseline. This is encouraging as it proved that the models are all working to outperform baseline, we just need to choose the best one.
- Both the train and test bloc results were consistent against baseline as well.

**MODEL SELECTED**

Based on the results, although all models performed better. Lasso Regression produced the best result of reducing variance, RMSE score while maintaining with 5% tolerance of R2. We also observed that there are 4 features that Lasso zeroed their coefficients which meant that we could removed these 4 features from the model.

The optimal alpha was calculated at 111.200979547034. Thus, we decided to fine-tune the Lasso model by removing the 5 features and re-run it at optimal alpha, then setting it as the selected production model.

## Re-run Lasso with reduced features

We then ran Lasso with the reduced features and plotted the predictors and their coefficients.


Based on the selected features and after going through regularization (constraint to the limits of the range of tests for alphas). The best predictors are depicted in the Lasso model that minimized the RMSE the most. This production model shall be the basis for predicting sale prices of the Kaggle's Test data file.

![pic10.png](attachment:pic10.png)

## Generating predictions for Kaggle's Test data

As the Kaggle's test data was unseen till now, we repeated the data cleaning and feature engineering processes to ensure that it was an apple to apple preparation.

We took special care to ensure that the columns sequence in both production model and the test file was the same. Otherwise, the application of scaling will not be uniform.

Thereafter, we train the production model once again with the Kaggle's Train data, transform the Kaggle's Test data and generated the predictions.

We matched and inserted back the `Id` column before exporting the final Kaggle submission file. We also took a quick look at the summary statistic of the submission data as a sensibility check, for example no negative sale price and the price range was not blown out of proportion.

![pic11-2.png](attachment:pic11-2.png)

## Kaggle Submission Result

After submitting to Kaggle, this was our significant improvement in result from the first submission.

![kaggle_submission-2.png](attachment:kaggle_submission-2.png)

## Conclusion and recommendations

**Conclusion**

In this project, we utilized feature elimination techniques such as EDA and correlation analysis. This helped to trim the 80 features down to 28 prior to model building.

Simple feature engineering was then employed based on logical assumption with limited domain knowledge. There were 123 features when stacked with those generated by hot-encoding. Using correlation analysis, we trimmed that down to 30 most significant features.

We passed that data through a baseline linear regression model to established a baseline score. Then attempted to regularize the other various models (Linear, Ridge, Lasso, ElasticNet) using model tuning to ensure that we had chosen the best model that could make the best predictions on unseen data.

In conclusion, we were able to reduce the measuring metric, RMSE using a Lasso regression model that greatly improved on the performance of predictions as compared to the baseline linear regression model. This improvement was eventually validated by a very much improved Kaggle submission score. Despite this improvement, we felt that there were room for improvement to search for an even better model.

**Recommendations**

Some recommendations for future enhancement include:
    - gaining more domain knowledge thru external research so as to understand the data better to make better inference
    - more in-depth analysis during data cleaning and EDA steps to ensure the correct treatment to NaN, Zeros and outliers
    - expanding on numbers of engineered features with improved domain knowledge
    - getting more data
    - increasing the range of alphas in the tuning process and even try to alter other hyperparameters besides just alpha
    - conducting survey with stakeholders (buyer/seller/realtors/tax agencies) to establish if the predictions and features are indeed strong and relevant features that may affect sale prices

Finally, the project started with the challenges faced by various stakeholders in using and interpreting 80 features to measure or predict housing prices in Ames, IA. Most of the time. they relied on their intuition or the factors of price known to them like location and size. This made it very difficult to assess and quantify the relationship between factors and prices. Thru the use of data analytics and machine learning techniques, several models were evaluated to determine the best set of predictor features and the best regression model to give the best prediction of prices. The overall goal metric of RMSE reduction was achieved as well.

Thank you.

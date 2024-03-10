**Regression - Retail Sales Prediction**

**Project Summary -**

Rossman Sales Prediction data contains historical sales data for a retail store chain. The data includes information about the store such as Competitior’s Detail, holiday’s, number of the customers, sale transaction's date and amount of sale on each day. We were tasked to forecast the "Sales" for the test set.

After understanding the data and getting variables, we first gathered and cleaned the data, handled the null values and finally for getting better results we merged two datasets after that we have also typecasted the needed features into required format by type casting in order to visualize them properly. We performed indepth EDA and plotted different types of graphs by separating them into univariate, bivariate and multivariate categories as a result, We came accross some meaningful insights that helped us to make future decisions of ML model pipeline. Then further on, using feature engineering and data preprocessing we have extracted new features like PromoDuration and CompetitionDuration with the help of some features which are not directly impacting to Sales. We also tried to get some impacting features by removing multicollinearity within the independent variables with the help of various inflation factor(VIF). Under the umbrella of feature engineering we have detected and treated the outliers with the help of IQR technique and capped all the outliers of continous features in 25-75 percentile. Also, we noticed that some of the features were categorical in nature and ML model can not understand the language of alphabets(strings).So, we have encoded them into numericals using One-Hot Encoding technique as they were unordered in nature.

In order to get normally distributed data we have applied various transformation techniques such as Logarithmic Transformation, Exponential Transformation, Square root Transformation and some others as well and plotted the quantie-quantile plot to visualize how far our data points are from the normal distribution.To scale the data We used the sklearn library StandardScaler.

Now as we are ready with our final features we splitted it into training and testing sets. Next, we choose some machine learning algorithms and use the training data to train the model. Finally, we evaluated the model's performance on the testing data to see how well it is able to predict the sales for the real time data. For this task we used many machine learning algorithms, such as linear regression, decision trees, random forests, LightGBM and XGboost. For the less complex models like Linear Regression, we could achieve the r2 score of 0.75 and accuracy i.e 100-MAPE of 93% even after applying regularisation techniques such as Lasso,Ridge and Elstic Net. In order to capture more variance and train our model more aptly, we decided to go for more complex models one by one. After training our datasets on decision trees, random forests, LightGBM and XGboost, we could gather the r2 score of 0.94 with the best accuracy of 97% using XGboost with mean absolute percentage 
error of only 2%. Also we got the mean of residuals as 0.0 which is indicating towards perfectly normally distributed residuals which is one of the good characteristics of good residual plot. From the above experiments and identifications, we have choosen the XGboost as our final optimal model among all 5 models for deployment as it is predicting the highest accuracy with the least error.

Overall, while building a machine learning model on Rossman Sales Prediction Data, we applied combination of data processing, machine learning techniques, and model evaluation skills. It was a challenging task and we faced some failures as well but with the right approach and knowledge, we successfully created a model that can accurately predict sales upto six weeks in advance!

**Problem Statement**

Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann store managers are tasked with predicting their daily sales for up to six weeks in advance. Store sales are influenced by many factors, including promotions, competition, school and state holidays, seasonality, and locality. With thousands of individual managers predicting sales based on their unique circumstances, the accuracy of results can be quite varied. You are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales" column for the test set. Note that some stores in the dataset were temporarily closed for refurbishment.


**Conclusion**

Exploratory Data Analysis (EDA) is an important step because it allows for the initial investigation of a dataset. It helps to identify patterns, anomalies, and relationships in the data, as well as to detect any potential issues such as missing values or outliers. EDA also helps to generate hypotheses and inform the development of more advanced modeling techniques, such as machine learning. Additionally, it is a good way to understand the data, which is crucial for good decision making. EDA also helps to provide a deeper understanding of the data and helps to guide the direction of further analysis. After performing EDA we have drawn the following conclusions:

Sales vs Customers graph shows positive correlation between 'Sales' and 'Customers'. As the number of customers increases, the sales also tend to increase.
December being a festive month attracts more sale than the rest of the months. Also, November has slightly more sales than other months. This could be due to the 'Black Friday' sale which is very popular across the globe. As Rossmann Stores deals in health and beauty products, it can be guessed that November and December sales are due to the celebratory nature of people who love to buy beauty/health products leading to the sudden increase in sales.
Day 1 and day 7 witness the highest sale indicating they are probably days falling on the weekend. Day 2 to day 6 generate medium to low sales indicating they are probably weekdays where customer footfall is low.
As stores are getting promoted, more sales are getting generated.
Plot between StateHolidays and sales shows that during Easter and Christmas holiday sales are actually high but for other holiday such as public ,sales are comparatively low.
Sales for the store type b is the highest . Store type B might be located in a more affluent or high-traffic area, which would increase the number of potential customers. Store type B may have a more favorable layout, which makes it more attractive to customers and makes it easier for them to find the products they want, resulting in more sales.
Sales are highest for the assortment b . This assortment may have a good mix of products that are in high demand or that are unique to the store, which would result in more sales.
We observed that mostly the competitor stores weren't that far from each other and the stores densely located near each other saw more sales.
We can conclude that Sales are high during the year 1900, as there are very few store were operated of Rossmann so there is less competition and sales are high. But as year pass on number of stores increased that means competition also increased and this leads to decline in the sales.
From plot sales and competition Open Since Month shows sales go increasing from Novemmber and highest in month December. This may be due to Christmas eve and New Year.
We can conclude from sales vs promo2 graph that customers are slightly less responsive to the stores(i.e sales) that are running consecutive promotions. One possibility could be customers might have already taken advantage of a similar promotion earlier. Another reason could be store might not have invested enough in promoting the promotion to customers, resulting in lower awareness and fewer sales. Also, if the store is running same promotion again and again, it could have resulted into lower customer footfal and ultimately leadind to fewer sales.
Sales vs Promo2SinceYear barplot explains that sales were still the highest when the store wasn't running any consecutive promotional events. But in 2014, the sales were really shoot up and they are recorded as 2nd highest. Good quality products, better deals, shutdown of competitions etc could be the reasons.
We can see the promo interval Jan, Apr, Jul, Oct records the 2nd highest sales as it marks the festive season. However, the other intervals are recording sales that are close to the 1st interval.

**Conclusions drawn from ML Model Implementation**


Close predictions of any ML model highly impacts the business growth. Before going to further model deployment one should have to check how accurately the model is predicting and performing with the real life data.

Conclusions drawn from any model are very helpful to identify wheather the model is fully baked and good to go for deployment process or needs further refinement. In this section first we will talk about some general points that are essential for every ML model and then will talk about the project oriented conclusions we made:

**General conclusions:**


The implementation of an ML model can greatly improve the performance and accuracy of a system or application.
It is important to carefully select and preprocess the data used for training and testing the model.
Regular evaluation and tuning of the model is necessary to ensure optimal performance.
The use of appropriate evaluation metrics can help to measure the performance of the model.
The integration of the model into the system or application should be done in a way that allows for easy maintenance and updates.
The ethical and legal considerations of the model's use should also be taken into account.
Project conclusions:

We have implemented various regression model started with Linear Regression and then we have tried other non linear models too. For each of the model we have tried to tune the hyperparameters as well in order to minimize the errors and drawn following conclusions

In Linear Regression we got the accuracy of ~93% and the model is capturing 75% of variance even after using regularization techniques that means our data is not perfectly linearly dependent with target variable(Sales).
For Decsion Tree we have achieved ~96.3% accuracy with maximum depth of 18 and on increasing the depth over it we are falling towards overfitting and MAPE of 3.6% which ultimately increases the mean absolute percentage error.
Giving preference to each of the variable always results in better accuracy as small subsets can provide significant accuracy percentage. Ensemble technique i.e Random Forest has given the accuray of ~96.96% with total trees of 100 in the forest with hyperparameter tuning.
We have also tried gradiant boosting technique with LightGBM although we got the similar results as Random Forest(~96.4%) but we got the more fast results as it has used all the cores and decrease processing time. While training the large dataset one should try LightGBM for good results in less time.
At last we have implemented our final model i.e XGboost and achieved the accuracy of 97% with mean absolute percentage error of only 2%. Also we got the mean and median of residuals at 0.09 which is indicating towards excellent residual plot.
Since, a good residual plot have the following characteristics:

It should contain high density of points close to the origin and a low density of points away from the origin.
It is symmetric about the origin.
The residual plot obtained from XGboost of fulfilling both the characteristics and from the above experiments and identifications we have choosen the XGboost as our final optimal model for deployment.

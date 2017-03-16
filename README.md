# AmazonTextMining
An exploratory analysis on amazon text data
Sentiment Analysis of Amazon Phone Reviews

Aravind Sridhar



Introduction

Text Mining is a type of sentiment analysis that finds patterns in text in order to build a model.  This paper utilizes text from Amazon cell phone reviews to train classification models that predict the star rating of a review.  Star ratings are integers ranging from one to five that represent how the reviewer feels about the product. This paper will perform an exploratory analysis of the dataset, describe the predictive task in detail, build classification models from the dataset, and interpret the results. Classification models like these can be applied to help meet the needs of customers to provide a better service.  In industry, understanding customer needs is often referred to as ‘Voice of the Customer’ analysis [1].  Classifying these Amazon reviews will face the challenge of training models with imbalanced datasets.  To model an unequal distribution of data successfully, two strategies described by Veni and Rani [2] will be implemented: Cost-Sensitive Learning and Performance Metrics.

*The Amazon phone reviews  were obtained from:
https://www.kaggle.com/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones.


Exploratory Analysis of Data


Figure 1: Sample Amazon Phone Review


The dataset used consists of 200,000 cell phone reviews that include each of the six following categories:
  
After randomly shuffling the reviews, a training and test set, each of 100,000 reviews, are created.  An initial exploration of the training data gives us basic information about each category:

Product Name:
String of Product Name
Mostly Android products (some Apple)
Brand Name:
String of manufacturer name
Price:
Float value of price
Some reviews do not have prices
Average price  = $226.347
Review Votes:
Mostly 0
Reviews:
Average length of text:	182.674 characters 
Many of the reviews have english stopwords (‘i’ , ‘the’, ‘it’, etc.)
Some reviews contain words that are fully capitalized

Figure 2: Example of text of fully capitalized words

Rating:
An initial exploration of the ‘Rating’ label returns interesting results.  The mean star rating is 3.7652 stars, showing that the ratings are skewed towards positive reviews, equating to high star ratings.  The training set’s star ratings have the following distribution:

Figure 3: The rating distribution of the training set

Star Rating
Percentage of Reviews
5
49.59 %
4
13.05 %
3
7.67   %
2
6.26   %
1
23.42 %
Table 1:  The percent distribution of star ratings of the training set

Immediately we see that a disproportionate amount of the ratings are five stars (almost 50%).  As previously mentioned, building a classifier for an imbalanced dataset can present additional challenges.  Much research suggests that an unequal class distribution tends to build biased classifiers for the majority class.  This can be harmful when the classification of infrequent, or minority, classes is valued more than that of the majority class.  Classifying imbalanced datasets is a popular topic of research in machine learning, as there are many methods for building the successful classifiers.  
.


Predictive Task

For this paper, the text from Amazon cell phone reviews will be analyzed to predict the star rating given by the reviewer.   It is important to acknowledge that using the ‘Product Name,’ ‘Brand Name,’ ‘Price,’ and ‘Review Votes’ features could also be used to improve the predictive model,however this paper focuses solely on text mining classification.  

Evaluating Models

In classification, not all models should be evaluated the same way.  Cohn, Jeni, and Torre maintain the notion that an imbalance in data distribution can yield misleading conclusions about that data [3].  They also argue that a broad range of performance metrics should be used to evaluate a model.  In order to evaluate star rating classification models, two performance metrics are used: Total Accuracy and Minority Accuracy.  The first metric, Total Accuracy, is defined as follows:

Metric 1: Total Accuracy = Number of Correct Predictions  /  Length of Test Set

In order to optimize for total accuracy, the classifier must correctly classify as many reviews in the test set as possible.  The number of correct reviews in the test set determines the accuracy of the classifier.  This is not always the best approach for classification.  Veni and Rani [2] argue that the total accuracy metric has significant flaws.  This approach minimizes overall error which values majority classification more heavily than minority classification in this dataset.  Furthermore, this approach assumes that the dataset is balanced with an equal distribution and that all errors have the same cost. There are many scenarios in which correctly classifying a minority class is much more important than optimizing total accuracy, such as: fraud detection, airport security, and detecting manufacturing defects.  For these situations, it is important to have a model that prioritizes the accuracy of predicting minority classes.  Thus, the second metric used to evaluate models will be Minority Accuracy:

Minority Classes = All classes not defined as Majority	
	
Minority Prediction = Number of correct predictions of the minority classes
 
Metric 2:  Minority Accuracy = Minority Prediction  /  Total number of Minority Classes in test set

For the Amazon cell phone review dataset, the majority class is a 5 star rating and the minority classes are the ratings in the set {1, 2, 3, 4}.  These two performance metrics will be compared and contrasted for different classification models.  

Preprocessing

Prior to building our model, there is preprocessing required to convert raw review text into a workable format. This preprocessing will use methods obtained from reference [9].  Firstly, all unicode characters are decoded to ascii characters to clean up any encoding bytes that appear in the text.  Secondly, each review (initially a large string) is converted into a list of all the words in that review while simultaneously converting each letter to lowercase.  Thirdly, all punctuation and empty strings are removed from the list.  Finally, all english stopwords (the, it, so, etc.) are removed and stemming is applied to each word.  Upon completion of this process, each review is represented by a list of lowercase, stemmed words with all stopwords and punctuation removed.
Baselines

To compare performance metrics of classification models, simple baseline models are established.  These baseline models do not attempt to solve the imbalanced dataset problem.  The first and most naive baseline model is one that predicts a five star rating for any input.  Although simplistic, this model is expected to have an accuracy around 50% due to the high frequency of 5 star ratings in the test set.  All classification models created through supervised learning should be an improvement of the baseline performance.

Baseline Model 1: Predict 5 every time
Total Accuracy: 54.14%
Minority Accuracy: 0%

For the second baseline model, the preprocessed review data is converted into a bag-of-words format using the CountVectorizer library in sklearn.  Each review is now represented by a vector that holds the count of each word in that review.  The second baseline will train a Multinomial Naive Bayes classifier using the bag-of-words format.  Multinomial Naive Bayes models are non-binary models which use text frequency (via bag-of-words in this model) for classification [4].


Figure 4: bag-of-words model using CountVectorizer

Baseline Model 2: Multinomial Naive Bayes with Bag-of-Words
Total Accuracy: 69.33 %
Minority Accuracy: 40.867 %

The third baseline will apply a TF-IDF (Term Frequency-Inverse Document Frequency) transformation on the preprocessed reviews instead of the bag-of-words transformation used previously.  TF-IDF assigns weight to each word in the review based on how unique it is throughout all reviews [5].  Ramos [10] describes advantages and limitations of TF-IDF:

Advantages: Efficient and simple algorithm, straightforward encoding
Limitations: Does not maintain relationships within words, ‘s’ can affect words - dog considered different from dogs

TF-IDF is especially strong when there are many reviews (robust document) [6] to compare. The same Multinomial Naive Bayes classifier is applied.

Baseline Model 3: Multinomial Naive Bayes with TF-IDF
Total Accuracy: 70.63 %
Minority Accuracy:  41.62 %

The final baseline model will allow the TF-IDF models to capture n-grams ranging from one to four words.  This helps to factor for the limitation described by Ramos by maintaining relationships between some words.




Figure 5: TF-IDF with n-grams

Baseline Model 4: Multinomial Naive Bayes with TF-IDF that contain (1-4)-grams
Accuracy: 77.00 %
Minority Accuracy: 54.97 %

Building a Classification Model

The proposed model will be a Random Forest Classifier or RFC.  Breiman [7] describes RFC’s as a combination of decision tree predictors that form a forest using decision functions.  The trees act as weak  independent learning models that combine to form a stronger model (forest) in a process called bagging.  When an RFC is trained, multiple decision trees are created by randomly and independently sampling the features. Trees compete via decision functions to vote for the most popular class.  These decision functions evaluate the correctness of trees to form a forest of the most accurate trees.  RFC’s can greatly improve classification accuracy. Horning [7] discusses the advantages and limitations of RFCs.

Advantages:
Computational Scalability for large dataset (fast)
Good at dealing with outliers
Takes care of overfitting through multiple decision trees
Not hyper sensitive to parameters used to train it
Limitations:
Not good at regression - it is not possible to predict beyond range of labels in training data

Since the RFC will be for classification and not regression, it is an acceptable model to use.  First, we train an RFC using the Random Forest Classifier library from sklearn, while still ignoring the imbalanced dataset problem.  The features are the same features obtained from the TF-IDF transformation in Baseline Model 4.


Figure 6: Sklearn Random Forest Classifier being Trained

Classification Model 1: Random Forest Classifier
Accuracy: 77.70 %
Minority Accuracy: 57.09 %

 *Classification Model will be shorted to CM

This model performs better than all previous baselines in total accuracy and minority accuracy.  To further optimize model for minority accuracy, the imbalanced data set must be accounted for.  There are many established ways to approach the imbalanced dataset problem.  In this paper, Cost-Sensitive Learning will be applied to our RFC to improve the Minority Accuracy performance metric.

Cost-Sensitive Learning

In cost-sensitive learning, a model is penalized a cost, or weight, when it misclassifies a minority class.  A famous example of a cost-sensitive approach is seen in airport security.  In this example, a security guard views the contents of suitcases as they pass through a security scanner and needs to binarily classify each suitcase as either safe or dangerous.  Barring unusual circumstances, a large majority of the suitcases scanned are safe.  However, the cost of overlooking a dangerous suitcase could have dire consequences.  Therefore, security guards choose to open any suitcase they believe to have a potential for being dangerous, although statistically the suitcase is most likely safe.

Minority Classes: 	{1 , 2 , 3 , 4}
Majority Class: 		{5}

In this paper, a cost-sensitive RFC applies controlled weights to minority classes in two steps.  First the class distribution spread is analyzed (figure 3).  Next, costs are applied to the minority classes to balanced this spread.  In our model, the majority set contains one class, {5}, that makes up 50% of the data set.  The minority classes, {1,2,3,4} must be an added weight to train the RFC.  The result is a weighted RFC which treats the classifying the minority classes with more importance than the majority Classification Model 2 applies the class-weight function from the Sklearn library.  The class-weight function allows control of weight distribution such that the minority classes can be given a heavier weight than the majority.  The Weighted RFC used in CM-2 is trained to apply a 22% importance weight on each of the majority class and a 12% importance weight on the majority .  This RFC treats minority classification with an 88% importance weight.


Figure 7: Class cost-sensitive RFC: labels 1 - 4 weighed more than label 5

Classification Model 2: Class-Weighted Random Forest Classifier
Accuracy:  77.59 %
Minority Accuracy:  57.097%

CM-2 performed worse in total accuracy than CM-1 due to overfitting the features of the minority classes and improved minority accuracy by .007%.  This shows that the class-weight function, by itself, did not help by much for this dataset. 

Similar to the class-weighting function, the next classifier will utilize the Sklearn sample-weighting function.  This approach assigns weights to all  feature array’s whose label belongs in a minority class.  In Classification Model 3, all features that correspond to a minority class are treated with five times the importance of those that belong to the majority.  This approach forces the classifier to learn “more” from the features that correspond to minority classes.


Figure 8: Sample cost-sensitive RFC: samples with labels 1 - 4 weighed 5 times more than samples with label 5

Classification Model 3:  Sample-Weighted Random Forest Classifier
Accuracy: 77.68 %
Minority Accuracy: 57.099 %

CM-3 has almost identical results to CM-2.  By itself, the sample-weight function (with the parameters shown in figure 8) is not that much of factor for this dataset

Combining Class and Sample Weighting

A final model is created which combines both the class weighting and sample weighting approaches together.


Figure 9: Combination of CM-2 and CM-3

Classification Model 4:  Sample-Weighted and Class-Weighted Random Forest Classifier
Accuracy: 77.51%
Minority Accuracy: 57.77 %

CM-4 presents a noticeable improvement of minority accuracy of  0.7%.  CM-4 also  resulted in the weakest total accuracy of all the Classification Models.

Issues

The biggest issue with having a cost sensitive approach was overfitting the RFC for the minority classes.  A cost sensitive approach improves the chances of correctly predicting minority classes but it also reduces the chances of correctly predicting a 5 star rating. This is why two different performance metrics were defined.  When total accuracy is used as the sole performance metric, CM-2, CM-3, and CM-4, performed worse than CM-1.  However, if minority accuracy is used as the sole performance metric, CM-2, CM-3, and CM-4, would be considered better classifiers.

Literature Analysis

Using Random Forest to Learn Imbalanced Data

Classifying imbalanced data is a strongly discussed field in academia.  The inspiration for using a Weighted Random Forest Classification came from Using Random Forest to Learn Imbalanced Data by Chen, Liaw, and Breiman [8].  They compare two different binary RFCs to analyze six imbalanced datasets - the Weighted RFC and a Balanced RFC. A balanced RFC utilizes sampling to artificially alter the class distribution so that each decision tree has an equal class representation.


Figure 10: Datasets Analyzed [8]

First, they establish many existing methods on the same data as the baselines: One-sided sampling, SHRINK, and SMOTE.  The results show that both Balanced RFCs  and Weighted RFCs improve the minority classification of the baselines.  While both RFCs have a considerably better minority accuracy than the baselines, they find it difficult to choose which is the winning minority classifier between the Balanced RFC and Weighted RFC.  They conclude that the model should be constructed depending on the data:
Balanced RFCs are more computationally efficient for large imbalanced data because only a small portion of the training set is used
Weighted RFCs need to use the entire training data so they are better off being used for smaller data sets

This results of the Weighted RFC for Amazon Cell Phone reviews confirm these results: minority accuracy is indeed improved with this model.  Based on the conclusion of Chien, Law, and Breiman, we find that it could have been more effective to use a Balanced RFC for the Amazon Phone Review data because of its size: 200,000 Reviews.

SMOTE for high-dimensional class-imbalanced data
Another state-of-the-art method dealing with classifying imbalanced data sets is the Synthetic Minority Oversampling Technique (SMOTE).  Blagus and Lusa [11] analyze the SMOTE technique two types of imbalanced data set - one with high dimensional data and the other with low dimensional data.  SMOTE is an oversampling method that uses existing minority class data to create synthetic minority class data. The problem with simple oversampling is that it tends to create overfitted models.  The synthetic data produced through  SMOTE attempts to factor for overfitting.  

Blagus and Lusa conclude that SMOTE is most effective on low dimensional data.  It can also be effective for high dimensional data K-NN classifiers.  However, non K-NN (nearest neighbor) classifiers created through SMOTE are ineffective and tend to have similar results to simple over sampling.  Because RFCs use a nearest neighbor algorithm, SMOTE might have been useful.

Results

When evaluating classifiers it is important to use different performance metrics in order to better understand them.  For our dataset, different features had different impacts total accuracy and minority accuracy.

Feature
Impact
TF-IDF representation with n-grams
Improved: Total Accuracy, Minority Accuracy
Random Forest Classification
Improved: Total Accuracy, Minority Accuracy
Weighted Random Forest Classification
Improved: Minority Accuracy
Worsened: Total Accuracy

Figure 11: Summary of Feature Impact

The best classifier for optimizing minority accuracy was CM-4 (weighted RFC) which classified 57.77 % of the minority classes correctly.  This model succeeded in improving minority accuracy because minority classes were given a weight in order to account for the unequally distributed dataset.  The best classifier for optimizing total accuracy was CM-1 (normal RFC) which classified 77.7 % of ratings correctly.  Minority accuracy might be improved if either an over-sampling technique such as SMOTE, or a Balanced Random Forest Classifier was used.  In addition, training an RFC that also incorporates the “Price” feature could have improved both total and minority accuracy.


Works Cited
Voice Of the Customer (VOC)
https://www.isixsigma.com/dictionary/voice-of-the-customer-voc/
C.V. KrishnaVeni,T. Sobha Rani ―On the Classification of Imbalanced Datasets‖ IJCST Vol . 2, SP 1, December 2011 
L. A. Jeni, J. F. Cohn, F. De La Torre, "Facing imbalanced data-recommendations for the use of performance metrics", Proc. Int. Conf. Affective Comput. Intell. Interaction, pp. 245-251, 2013
A. McCallum and K. Nigam. A comparison of event models for naive bayes text classification. In AAAI’98 Workshop on Learning for Text Categorization, Madison, Wisconsin, 1998.
Zhang, Y.T., Gong, L., Wang, Y.C., 2005. An improved TF-IDF approach for text classification. J. Zhejiang Univ.-Sci., 6A(1):49–55. [doi:10.1631/jzus.2005.A0049] 
http://lib.dr.iastate.edu/cgi/viewcontent.cgi?article=1215&context=cs_techreports
Breiman, L. Machine Learning (2001) 45: 5. doi:10.1023/A:1010933404324
Using Random Forest to Learn Imbalanced Data (2004) by Chao Chen, Andy Liaw, Leo Breiman
https://de.dariah.eu/tatom/preprocessing.html 
J. Ramos. Using TF-IDF to Determine Word Relevance in Document Queries. Technical report, Department of Computer Science, Rutgers University, 2003.
Blagus R, Lusa L (2013) SMOTE for High-Dimensional Class-Imbalanced Data. BMC Bioinformatics 14(106)

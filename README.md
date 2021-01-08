# Dialect identification

## Background

This project is a practice in automated Dialect Identification – a subtler subset of Language Identification. Dialect Identification is a vivid research topic, since the techniques have many valuable applications. For instance, while location information is often used for review/tweet analysis, recognising nuanced differences in dialect could offer deeper insight into the individual behind the text - it may contain implicit information on where the individual grew up and the many social implications of that, what social group the individual affiliates with, etc.

As a basis for exploring the application of machine learning to the DI issue, I used lyrics from East and West coast (USA) rappers. Here, the idea is that the language used by east and west coast rappers differs in recognisable ways owing to a kind of difference in dialect between the two coasts. It is specifically this difference which I want my models to be using to make predictions – not any idiosyncrasies in individual rappers’ lyrical content.

I apply natural language processing (NLP) techniques using the NLTK module, and use several classification techniques.

## Problem Statement

*The goal of this project is to accurately classify songs from east and west coast rappers by their respective coast, using only information which can be described as relating to the differences in dialect between coasts.*

## Method Overview

In order to truly achieve a model which uses information associated with east/west dialects, extensive use of stopwords were required. Including names of cities or areas in the tokenised predictors would lead to superior accuracy scores, but would mean that the model would not truly be faithful to the issue of dialect identification. Similarly, rappers spend much energy referencing themselves - information which also will lead to 'cheating' in the sense that were the model to be trained on this information, it would be performing the classification based on rapper-specific information as opposed to dialect information. 

A good way to ensure that models are truly performing a classification based on non-rapper-specific language traits is to split the training and test sets by rappers, stratifying on the target variable. In this way, any 'cheating' (or over-fitting) going on in the training stage in terms of rapper-specific information will not carry through to the testing stage. This would not be the case were we to allow the model to train itself on songs from all rappers, and test itself on other songs from the same rappers.

Therefore, being faithful to the problem required a custom approach to the classification problem, since scikit-learn’s GroupKFold has no option to stratify the data splitting based on class. A customised train-test-split function where the splitting was done for groups was required.


## Data

First on the agenda was actually getting the lyrics to a bunch of tracks from east/west coast rappers. In an effort to limit within the data an evolution of the rap-dialects, I use only tracks released between 1987 and 2000 (inclusive). As such, changes in vocabulary/style over time within each group (east/west) will not play much of a role. The boundaries of the window are not especially important, but were chosen to contain the years in which the east-west coast conflict/rivalry was prominent. 

The data collection process looked like this:

**1:** Get the names of west and east coast rappers from a Wikipedia list. This list was far too broad, and required filtering down to the rappers who are actually rappers, and who actually match the aforementioned location criteria.  
**2:** Use Spotify API's information on these rappers to determine whether they align with appropriate genres. For example, drop rappers falling under Christian Rap and Trap Queen etc.  
**3:** Use Spotify API to get all the remaining rappers’ track listings for albums released between 1987 and 2000.  
**4:** Use the Genius API to search the artists and take the names as they appear within Genius.  
**5:** Use the artist names under Genius' format to scrape their lyrics from the Genius website.  

## Data Cleaning

There were some things scraped from Genuis which were not what I was looking for. For example, 'interlude'-type songs, where there were only a few lines of conversation, or a snippet from a tv broadcast. Also, there were some 'lyrics to appear soon' entries which we need to drop from the dataset. To deal with these recurring issues, I dropped all songs where the total number of characters in the lyrics were less than 300.

Then, I put the lyrics into a format which was EDA-friendly, by dropping new line characters, words with digits in them, and by dealing with punctuation.

## Exploratory Data Analysis

A preliminary exploration of the dataset showed a significant class imbalance, with a greater number of rappers songs belonging to the West coast (1511:1179 songs; 48:37 rappers).  Also, there was much variation in the number of tracks belonging to different rappers in the dataset. These features of the dataset, along with its relatively small size, are not ideal. We would like an even distribution of tracks across artists so that no one artist is overly prominent in the training of the model, and thus the model is not overly sensitive to a small portion of the rappers' lyrical habits. If I were to extend this project, I would attempt to deal with this, perhaps by repeating entries for under-represented rappers.  

Exploring the lyrical composition of the tracks was interesting. Extracting features such as the numer of unique words in a song as a portion of the total word count provided some cautious evidence towards rumours that the east coast rappers are more 'lyrical' than their west coast rivals. These features also provided a very useful way of detecting tracks which needed to be dropped for similar reasons to those previously discussed.

## Stopwords

As previously outlined, remaining true to the goals of this project required extensive use of rapper- and location-specific stopwords. Ammassing the full list of stopwords was an iterative process. There were many clear 'cheat' words that initially showed up in the strongest logistic regression coefficients, which were added to the list. Some tools used for EDA also proved to be very useful.  

## Modelling:

### Custom train-test-split:

Running classification models using scikit-learn's standard train-test-split function yields very strong accuracy - immediate mean cross-validation score of 84%. However, as previously explained, these results are unfaithful to the dialect identification objective of the project. So, I created a custom function for the train-test-split, which made use of the random module to cumulatively fill the test set with rappers' entire track list, at every stage choosing a random rapper from whichever coast was underrepresented in the test set as compared to the overall class imbalance of 0.57 until the size of the test set exceeded a quarter of the total number of observations - thus achieveing stratification with a grouped train-test-split. The function allowed random state selection.

This function complicated matters for the rest of the modelling. Firstly, the custom mode of splitting the data had to be incorporated in any cross-validation. While scikit-learn's cross-validation function allows custom splitting methods, it didn't work with my custom method, because of the added stratification requirement. Thus, I also built a further custom cross-validation function. For the splitting into folds, an input was the list of rappers in the training set, from which it used a similar technique to the initial train-test-split to split the training set into the specified number of folds, attempting to keep the fold sizes and class proportions as even as possible.

From here, the custom cross-validation method could not be used as a custom scorer in scikit-learn's GridSearchCV function. For this function, a custom scorer has to only take (estimator, X, y) as inputs. My c-v function required the extra input mentioned. So, a custom method of comparing model performances was also needed. Luckily, GridSearchCV is essentially just a simple for loop across the specified models and hyperparameters, and so this was not especially difficult.

### NLP:

To translate the lyrical text into a format which can be understood by the classifiers, I used scikit-learn's CountVectorizer and TfidfVectorizer. The CountVectorizer performs a simple bag-of-words transformation on the dataset: a column is created for each word that is used in the lyrics used for fitting, and each row entry in that column corresponds to the number of times that word appears in the corresponding track. The TfidfVectorizer is more complicated. It constructs a similar-looking matrix to the CountVectorizer, but rather than simply counting the number of appearances of a word in a track, it returns a number which depends positively on the number of times that word appears in a song, but which also depends negatively on the frequency of use for that word in other songs. Thus, it gives a sense of the *discriminating power* of a word between data entries.

I also tried using NLTK's PorterStemmer and SnowballStemmer, which provide different ways to trim a word into its core form in order that slightly different forms of a root word are recognised as the same variable. The most simple such transformation would be to get rid of the 's' at the end of a word where that 's' simply makes it plural.

### Results:

It is worth noting that given the relatively small dataset in terms of entries and in terms of the number of rappers behind the entries, model performances varied non-trivially between random states in the train-test-split and cross-val-split methods. Therefore for each model and hyperparameter set tested, I took the average scores across three random states to give a more representative account of model performance. It is also worth noting that for these averages, the test scores were consistently significantly higher than the cross-validation scores. This can probably be explained by the amount of data which the model is able to train itself on, and thus indicates that results on new data after training on all the data available would yield better results than the cross-validation stage indicates.

The models I used were Logistic Regression, K Nearest Neighbours, Random Forest, Bagging, Support Vector Classification, AdaBoost, Gradient Boosting, and a Multi-layer Perceptron (MLP).

The best model in terms of averages across the three mean cross-validation accuracy scores (one for each random state) was the MLP, with a score of 77.6%. The model's mean test score was 80.8%. Observing the classification report for this model, there was generally far stronger classification of east coast songs. A high precision (& low recall) of west coast songs, combined with a high recall of east coast songs, indicates that the model was quite trigger-happy with classifying songs as east coast. This is likely due to the class imbalance present in the data, and is an area which I would address if I were to extend this project.

It was also interesting to see which rappers' songs were frequently misclassified, making these rappers' lyrical style fall further outside of the coastal dialect as it is understood by the model. Another useful extension to this project might be to investigate these areas of frequent missclassification, hoping to find insights into where the model's understanding might be improved.

Given the non-parametric process behind MLP, there isn't much intuition to be gained into how this model performs its classification. To understand some differences between the two coastal dialects which drive the models, I investigated the strongest Logistic Regression coefficients within the classification. A positive coefficient here indicated an stronger association of that word with west coast rappers, and the larger the size of the coefficient, the stronger the association. The coefficients determine the model's perceived probability that a song is of either coast. As an example, 'word' has a strong negative coefficient, and is thus strongly indicative of an east coast song. A little research into some basic lyrical differences between rappers of the two coasts supports this association - 'word', short for 'word is bond', is a favourite piece of slang amongst east coast rappers.

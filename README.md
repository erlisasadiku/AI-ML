# BBC News Text Classification

This project implements a text classification pipeline to categorize BBC news articles into five categories:
tech, business, sport, entertainment and politics. The aim is to compare the performance of Logistic Regression and Naive Bayes models for text classification.

## Dataset
Is a public dataset of BBC News articles, contains 2225 labeled articles
- **Columns**:  
  - `category` → target label (five categories)  
  - `text` → raw news article text 

## Data Preparation
- Loaded dataset and explored basic statistics
- Visualized category distribution (balanced dataset)
- Checked for missing values (none found)
- Encoded categories into numerical format
- Preprocessed text:
    - Lowercasing
    - Removing punctuation and numbers
    - Stopword removal (NLTK)
    - Word stemming (Porter Stemmer)

## Feature Engineering
- Used TF-IDF Vectorizer with:
- Maximum 5000 features
- Unigrams + bigrams (ngram_range=(1,2))

## Model Training and Evaluation
- Logistic Regression (max_iter=1000)
- Naive Bayes
- Both models were trained on 75% of the dataset and evaluated on 25%.
- Evaluation metrics: accuracy, precision, recall, F1-score
- Confusion matrices were plotted for both models.

## Results
| Model                | Accuracy |  
|-----------------------|----------|  
| Logistic Regression   | **0.978** |  
| Naive Bayes           | 0.975    |

## Example Prediction
```python
text = "The government announces new economic policies."
print(predict_category(logReg, text, vectorizer, encoder))
# Output: "business"
```
Logistic Regression showed slightly better overall performance.
Confusion matrices and classification reports confirm strong results across all categories.
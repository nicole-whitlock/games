# Sentiment Analysis on Video Game Reviews

# Research and Selection of Methods
### Objectives
Our primary objective is to design and implement a robust pipeline that processes raw textual data, extracts meaningful features, and employs state-of-the-art classification algorithms to gauge sentiment accurately. This approach aims to provide actionable insights that can influence game development, marketing strategies, and overall industry decision-making.

The central challenge lies in interpreting the varied and often nuanced language found in game reviews. Reviews may include slang, domain-specific jargon, and subtle emotional cues that complicate sentiment classification. Overcoming these hurdles involves rigorous data cleaning, text normalization, and feature extraction. Our solution will harness both traditional machine learning and modern deep learning techniques to achieve high accuracy in sentiment analysis.


### Literature Review
### Benchmarking

Model for distilbert with 50000 samples and no additional layers
### Preliminary Experiments
# Model Implementation
### Framework Selection
### Dataset Preparation
Transfomers are well known for being able to capture the context and dependencies in words, and perform well with minimal data cleaning. It is generally not required to do extensive data cleaning on transformers so minimal cleaning was done:

- Removed leading, trailing, and extra whitespaces

Tokenization:

- `Distilbert-base-uncased` tokenizer was loaded and implemented to tokenize the text data for model training
- Function `tokenize_optimized` to tokenize data in batches to avoid memory crashing on the computer.
  - `MAX_LEN` set to 128
  - `truncation` = True to truncate longer strings of texts
  - `padding` = 'max_length' to add padding to the end of text strings that are shorter than the set max_length
  - `attention_masks` = True, to distinguish between padded and real tokens
  - `special_tokens`= True, to add tokens [CLS, SEP] to distinguish each sequence for classification
- Function outputs the input_ids and attention_masks for each row of the quote column in the dataframe

Data Splitting

- Data was split using `train-test-split` to first split data into training and test data and then the training data was further split to get validation data.

Sentiment Encoding:

- Sentiment was split into 'positive', 'negative', and 'neutral'. These values were then encoded into numerical values for training the model:

  ```
  sentiment_mapping = {
      'negative': 0,
      'neutral': 1,
      'positive': 2
  }
  ```



Dataset creation:

The model was created using a distilbert model with additional layers added on to try and avoid overfitting and improve performance of the model. The additional layers that were added need the data to be in a dataset format with input_ids and attention_masks that can be passed into the distilbert layer and then output to the next layer:

```
train_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': input_ids, 'attention_mask': attention_mask},
    train_labels
))
train_dataset = train_dataset.shuffle(100).batch(16)
```

```
val_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': val_input_ids, 'attention_mask': val_attention_mask},
    val_labels
))
val_dataset = val_dataset.batch(16)
```


### Model Development
### Training and fine-tuning
### Evaluation and Metrics

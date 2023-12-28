**Repository Overview:**

This repository houses classes essential for training and deploying a Recurrent Neural Network (RNN) to classify categories. It includes specialized classes designed for handling both Arabic and long English sentences.

**Arabic Data Processing:**

- **AbstractArabicDataProcessor:**
  All Arabic data processing classes extend this abstract class. It provides essential fields and methods for converting Arabic data into tensors suitable for machine learning. The only abstract method is `word_to_tensor`, where different approaches were explored for Arabic word processing.

- **SimpleArabicDataProcessor:**
  Utilizes the one-hot encoding method in the `word_to_tensor` method.

- **PositionEmbeddingArabicDataProcessor:**
  Extends one-hot encoding and assigns a value of 2 for characters in a word that signify specific positions. Classifications include:
  - If the second letter is "ا", it indicates a subject scale for 3-lettered verbs.
  - If the first letter is "م", it suggests an object scale for 3-lettered verbs.
  - Presence of "kasrah" diacritic indicates a subject scale for 4 or more lettered verbs.
  - Presence of "fatha" diacritic indicates a subject scale for 4 or more lettered verbs.

**Sentence Processing:**

- **SentenceProcessor:**
  Extends SimpleArabicDataProcessor and processes long English sentences. It converts sentences to a single word by removing irrelevant words, applies NLTK library stopwords, and uses the `word_to_tensor` method.

**RNN Model:**

- **RNN:**
  The core recurrent neural network, based on the tutorial [here](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html).

**Training:**

- **TrainingProcess:**
  Refactors the training process from the tutorial into a class. By instantiating and calling, it initiates the RNN training process using the SimpleArabicDataProcessor. For different processors, call `set_model` before instantiation.

**User Interface:**

- **user_interface:**
  Sets up the necessary code for user interaction. Steps include setting the list of files containing data (ensure filenames match categories), and modifying the data processor by adding the relevant code snippet after initializing the trainer class.

```python
# Example for using SentenceProcessor
model = SentenceProcessor(notion_files)
trainer.set_model(model)

# Or, for PositionEmbeddingArabicDataProcessor
model = PositionEmbeddingArabicDataProcessor(notion_files)
trainer.set_model(model)

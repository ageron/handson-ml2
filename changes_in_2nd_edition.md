# Changes in the Second Edition
The second edition has six main objectives:

1. Cover additional ML topics: additional unsupervised learning techniques (including clustering, anomaly detection, density estimation and mixture models), additional techniques for training deep nets (including self-normalized networks), additional computer vision techniques (including Xception, SENet, object detection with YOLO, and semantic segmentation using R-CNN), handling sequences using CNNs (including WaveNet), natural language processing using RNNs, CNNs and Transformers, generative adversarial networks
2. Cover additional libraries and APIs: Keras, the Data API, TF-Agents for Reinforcement Learning, training and deploying TF models at scale using the Distribution Strategies API, TF-Serving and Google Cloud AI Platform. Also briefly introduce TF Transform, TFLite, TF Addons/Seq2Seq, TensorFlow.js and more.
3. Mention some of the latest important results from Deep Learning research.
4. Migrate all TensorFlow chapters to TensorFlow{nbsp}2, and use TensorFlow's implementation of the Keras API (called tf.keras) whenever possible, to simplify the code examples.
5. Update the code examples to use the latest version of Scikit-Learn, NumPy, Pandas, Matplotlib and other libraries.
6. Clarify some sections and fix some errors, thanks to plenty of great feedback from readers.

Some chapters were added, others were rewritten and a few were reordered. The following table shows the mapping between the 1^st^ edition chapters and the 2^nd^ edition chapters:

|1^st^ ed. chapter  | 2^nd^ ed. chapter | %Changes | 2^nd^ ed. title
|--|--|--|--|
|1|1|<10%|The Machine Learning Landscape
|2|2|<10%|End-to-End Machine Learning Project
|3|3|<10%|Classification
|4|4|<10%|Training Models
|5|5|<10%|Support Vector Machines
|6|6|<10%|Decision Trees
|7|7|<10%|Ensemble Learning and Random Forests
|8|8|<10%|Dimensionality Reduction
|N/A|9|100% new|Unsupervised Learning Techniques
|10|10|~75%|Introduction to Artificial Neural Networks with Keras
|11|11|~50%|Training Deep Neural Networks
|9|12|100% rewritten|Custom Models and Training with TensorFlow
|Part of 12|13|100% rewritten|Loading and Preprocessing Data with TensorFlow
|13|14|~50%|Deep Computer Vision Using Convolutional Neural Networks
|Part of 14|15|~75%|Processing Sequences Using RNNs and CNNs
|Part of 14|16|~90%|Natural Language Processing with RNNs and Attention
|15|17|~75%|Autoencoders and GANs
|16|18|~75%|Reinforcement Learning
|Part of 12|19|~75% new|Productionizing TensorFlow Models

More specifically, here are the main changes for the 2nd edition (other than clarifications, corrections and code updates):

* Chapter 1 – The Machine Learning Landscape
  * Added more examples of ML applications and the corresponding algorithms.
  * Added a section on handling mismatch between the training set and the validation & test sets.
* Chapter 2 – End-to-End Machine Learning Project
  * Added how to compute a confidence interval.
  * Improved the installation instructions (e.g., for Windows).
  * Introduced the upgraded `OneHotEncoder` and the new `ColumnTransformer`.
  * Added more details on deployment, monitoring and maintenance.
* Chapter 4 – Training Models
  * Explained the need for training instances to be Independent and Identically Distributed (IID).
* Chapter 7 – Ensemble Learning and Random Forests
  * Added a short section about XGBoost.
* Chapter 9 – Unsupervised Learning Techniques (new chapter)
  * Clustering with K-Means, how to choose the number of clusters, how to use it for dimensionality reduction, semi-supervised learning, image segmentation, and more.
  * The DBSCAN clustering algorithm and an overview of other clustering algorithms available in Scikit-Learn.
  * Gaussian mixture models, the Expectation-Maximization (EM) algorithm, Bayesian variational inference, and how mixture models can be used for clustering, density estimation, anomaly detection and novelty detection.
  * Overview of other anomaly detection and novelty detection algorithms.
* Chapter 10 – Introduction to Artificial Neural Networks with Keras (mostly new)
  * Added an introduction to the Keras API, including all its APIs (Sequential, Functional and Subclassing), persistence and callbacks (including the `TensorBoard` callback).
* Chapter 11 – Training Deep Neural Networks (many changes)
  * Introduced self-normalizing nets, the SELU activation function and Alpha Dropout.
  * Introduced self-supervised learning.
  * Added Nadam optimization.
  * Added Monte-Carlo Dropout.
  * Added a note about the risks of adaptive optimization methods.
  * Updated the practical guidelines.
* Chapter 12 – Custom Models and Training with TensorFlow (completely rewritten)
  * A tour of TensorFlow{nbsp}2.
  * TensorFlow's lower-level Python API.
  * Writing custom loss functions, metrics, layers, models.
  * Using auto-differentiation and creating custom training algorithms.
  * TensorFlow Functions and graphs (including tracing and autograph).
* Chapter 13 – Loading and Preprocessing Data with TensorFlow (new chapter)
  * The Data API
  * Loading/Storing data efficiently using TFRecords.
  * Writing custom preprocessing layers, using Keras preprocessing layers, encoding categorical features and text, using one-hot vectors, bag-of-words, TF-IDF or embeddings.
  * An overview of TF Transform and TF Datasets.
  * Moved the low-level implementation of the neural network to the exercises.
  * Removed details about queues and readers that are now superseded by the Data API.
* Chapter 14 – Deep Computer Vision Using Convolutional Neural Networks
  * Added Xception and SENet architectures.
  * Added a Keras implementation of ResNet-34.
  * Showed how to use pretrained models using Keras.
  * Added an end-to-end transfer learning example.
  * Added classification and localization.
  * Introduced Fully Convolutional Networks (FCNs).
  * Introduced object detection using the YOLO architecture.
  * Introduced semantic segmentation using R-CNN.
* Chapter 15 – Processing Sequences Using RNNs and CNNs
  * Added an introduction to Wavenet.
  * Moved the Encoder–Decoder architecture and Bidirectional RNNs to Chapter 16.
* Chapter 16 – Natural Language Processing with RNNs and Attention (new chapter)
  * Explained how to use the Data API to handle sequential data.
  * Showed an end-to-end example of text generation using a Character RNN, using both a stateless and a stateful RNN.
  * Showed an end-to-end example of sentiment analysis using an LSTM.
  * Explained masking in Keras.
  * Showed how to reuse pretrained embeddings using TF Hub.
  * Showed how to build an Encoder–Decoder for Neural Machine Translation using TensorFlow Addons/seq2seq.
  * Introduced beam search.
  * Explained attention mechanisms.
  * Added a short overview of visual attention and a note on explainability.
  * Introduced the fully attention-based Transformer architecture, including positional embeddings and multi-head attention.
  * Added an overview of recent language models (2018).
* Chapters 17 – Representation Learning and Generative Learning Using Autoencoders and GANs
  * Added convolutional autoencoders and recurrent autoencoders.
  * Add Generative Adversarial Networks (GANs), including basic GANs, deep convolutional GANs (DCGANs), Progressively Growing GANs (ProGANs) and StyleGANs.
* Chapter 18 – Reinforcement Learning
  * Added Double DQN, Dueling DQN and Prioritized Experience Replay.
  * Introduced TF Agents.
* Chapter 19 – Training and Deploying TensorFlow Models at Scale (mostly new chapter)
  * Serving a TensorFlow model using TF Serving and Google Cloud AI Platform.
  * Deploying a Model to a Mobile or Embedded Device using TFLite.
  * Using GPUs to Speed Up Computations.
  * Training models across multiple devices using the Distribution Strategies API.

## Migrating from TF 1 to TF 2
Migrating from TensorFlow 1.x to 2.0 is similar to migrating from Python 2 to 3. The first to do is... breathe. Don't rush. TensorFlow 1.x will be around for a while, you have time.

* You should start by upgrading to the last TensorFlow 1.x version (it will probably be 1.15 by the time you read this).
* Migrate as much of your code as possible to using high level APIs, either tf.keras or the Estimators API. The Estimators API will still work in TF 2.0, but you should prefer using Keras from now on, as the TF team announced that Keras is preferred and it's likely that they will put more effort into improving the Keras API. Also prefer using Keras preprocessing layers (see chapter 13) rather than `tf.feature_columns`.
* If your code uses only high-level APIs, then it will be easy to migrate, as it should work the same way in the latest TF 1.x release and in TF 2.0.
* Get rid of any usage of `tf.contrib` as it won't exist in TF 2.0. Some parts of it were moved to the core API, others were moved to separate projects, and others were not maintained so they were just deleted. If needed, you must install the appropriate libraries or copy some legacy `tf.contrib` code into your own project (as a last resort).
* Write as many tests as you can, it will make the migration easier and safer.
* You can run TF 1.x code in TF 2.0 by starting your program with `import tensorflow.compat.v1 as tf` and `tf.disable_v2_behavior()`.
* Once you are ready to make the jump, you can run the `tf_upgrade_v2` [upgrade script](https://www.tensorflow.org/beta/guide/upgrade).

For more details on migration, checkout TensorFlow's [Migration Guide](https://www.tensorflow.org/beta/guide/migration_guide).

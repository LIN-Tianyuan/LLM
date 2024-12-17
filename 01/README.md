# Background knowledge

## 1. Large Language Model (LLM) Background

The Large Language Model (LLM) is an artificial intelligence model designed to understand and generate human language. Large Language Model
LLMs can handle a wide range of natural language tasks, such as text categorization, question and answer, translation, conversation, etc.

![Elovutionary Tree](/img/1.png)

In general, a large language model (LLM) is a language model that contains hundreds of billions (or more) of parameters (currently, models with more than 10B parameters are defined as large language models), which are trained on large amounts of textual data, such as the models GPT-3, ChatGPT, PaLM, BLOOM, and LLaMA.
These parameters are trained on large amounts of textual data, such as models GPT-3, ChatGPT, PaLM, BLOOM, and LLaMA.

As of 2023, language model development has gone through three phases:

Phase 1 : Designing a series of self-supervised training objectives (MLM, NSP, etc.), designing novel model architectures (Transformer), following the
Pre-training and Fine-tuning paradigms. Typical representatives are BERT, GPT, XLNet, etc;

Phase 2 : Gradually expand the model parameters and training corpus size, and explore different types of architectures. Typical representatives are BART, T5, GPT-3 and so on;

Phase 3 : Towards the era of AIGC (Artificial Intelligent Generated Content), model parameter scale steps into trillions of dollars, model architecture
for autoregressive architecture, large models towards the era of conversational, generative, multimodal, more focus on aligning with human interaction to achieve reliable, safe,
non-toxic models. Typical representatives are InstructionGPT, ChatGPT, Bard, GPT-4 and so on.

## 2. Language Model (LM)
Language Model aims to model the probability of generating lexical sequences, to improve the level of linguistic intelligence of machines, and to enable machines to simulate human speech and writing patterns for automatic text output.

The purpose of a language model is to model the probability of generating a sequence of words, so that the machine can simulate human speech and writing patterns for automatic text output.

Common understanding: A model used to calculate the probability of a sentence, that is, the probability of determining whether a sentence is human speech.

Standard definition: For a sequence of sentences, e.g. S = {W1, W2, W3, ..., Wn}, a language model calculates the probability of the sequence occurring, i.e. P(S). If the given
sequence of words is idiomatic, it gives a high probability, otherwise it gives a low probability.

Example:
 - Suppose we want to create a language model for Chinese, $V$ denoting the dictionary, $V$ = {我, 来, 学习} , $W_i$ belonging to $V$. Language Model Description:
Given a dictionary $V$, we can compute the probability $P(S)$ that any sequence of words $S = W_1,W_2,W_3, ...,W_n$ is a sentence , where $P >= 0$.
 - So how to calculate the probability $P(S)$ of a sentence? The simplest way is to count, assuming that there are a total of $N$ sentences in the dataset, we can count the number of times each sentence $S = W_1,W_2,W_3,...,W_n$ appears in the dataset, if we assume that $n$. Then $P(S) = {\frac{n}{N}}$. We can imagine that the predictive power of this model is almost zero. Once a word sequence has not appeared in the previous data set, the output probability of the model is 0, which is obviously quite unreasonable.
 - According to the chain rule in probability theory, the model can be expressed as follows:
 $$P(S) = P(W_1, W_2,..., W_n) = P(W_1) * P(W_2|W_1) * ... * P(W_n|W_1,W_2,...,W_{n-1})$$

If one can compute $P(W_n|W_1,W_2,...,W_{n-1})$, then one can easily get $P(W_1,W_2,...,W_n)$, so in some literature we can also see
to another definition of a language model: a model that can compute $P(W_1,W_2,...,W_n)$ is a language model.

From the text generation point of view, a language model can also be defined in such a way that, given a phrase (a word group or a sentence), the language model can generate (predict) the next word.

Based on the development of language modeling techniques, language models can be classified into four types:
 - Rule-based and statistical language models
 - Neural language models
 - Pre-trained language models
 - Large Language Models

### 2.1 Rule-based and statistical language modeling (N-gram)
Modeling and analysis of fixed-length text window sequences by manually designing features and using statistical methods is also known as N-gram language modeling. In the example above, the probability of a sentence sequence is calculated using the chain rule, which has two drawbacks:

 - The parameter space is too large: there are too many possibilities $P(W_n|W_1,W_2,...,W_n)$ for conditional probabilities to be estimated, and they are not always useful.

 - The data is sparse: there are many combinations of word pairs that do not appear in the corpus, and the probability based on the maximum likelihood estimation is zero.

In order to solve the above problem, Markov's hypothesis is introduced: the probability of an occurrence of an arbitrary word is related to only a finite number of words or words that occur before it.
 - If the occurrence of a word is independent of the words around it, then we call it a unigram, or a unitary language model.
 
 ![4](/img/4.png)

 - If the occurrence of a word depends only on the occurrence of a word before it, then we call it bigram.

 ![5](/img/5.png)
 - If the occurrence of a word depends only on the two words appearing before it, then we call it trigram.
 
![6](/img/8.png)
 - In general, the N-tuple model is the assumption that the probability of occurrence of the current word is only related to the N-1 words preceding it, and these probability parameters are all computable from large-scale corpora, such as the ternary probability:

  ![5](/img/10.png)

The most used in practice are bigram and trigram, and the next example is the bigram language model to understand how it works:
 - First we prepare a corpus (simply understood as a dataset that allows the model to learn), in order to calculate the parameters of the corresponding binary model, i.e. $P(W_i|W_{i-1})$, we have to count, i.e. $C(W_{i-1},W_i)$, then count $C(W_{i-1})$, and then divide can be used to get the probability.
 - $C(W_{i-1},W_i)$ The counting results are as follows：
 ![count1](/img/2.png)
 - $C(W_{i-1})$ The counting results are as follows：
 ![count2](/img/3.png)
 - So how does the bigram language model realize the results of parameter calculation for the above corpus? Suppose, I want to calculate think $P(_想|_我)≈ 0.38$ , the calculation process is shown as follows: (other parameters calculation process is similar)
 $$P(_想|_我)≈ {\frac{C(_我,_想)}{C(_我)}} = {\frac{800}{2100}} ≈0.38 $$
 - If the binary model (bigram) for this corpus is built, our target computation can be realized.
 - Calculate the probability of a sentence, as an example:
  ![count3](/img/6.png)
 - Predict the most likely next word in a sentence, e.g., 我想去打 [mask]? Think: mask = 篮球 or mask = 晚饭.
 ![count4](/img/7.png)
 - It can be seen that P(我想去打篮球) > P(我想去打晚饭), so mask = 篮球, contrasting with the real context and also with human habits. 

Characteristics of the N-gram language model:
 - Advantages: using great likelihood estimation, parameters are easy to train; completely contains all the information of the first n-1 words; highly interpretable, intuitive and easy to understand.
 - Disadvantages: lack of long-term, can only model the first n-1 words; as n increases, the parameter space grows exponentially; sparse data, inevitably OOV
problems; based solely on statistical frequency, poor generalization ability.

### 2.2 Neural network language model
Along with the development of neural network technology, people began to try to use neural networks to build language models and then solve the problems of N-gram language models.

![img11](/img/11.png)

The above figure belongs to one of the most basic neural network architectures:

 - The input to the model: $w_{t-n+1}, ... , w_{t-2}, w_{t-1}$ is the first n-1 words. Now we need to predict the next word $w_t$ based on these known n-1 words.
$C(w)$ denotes the corresponding word vector $w$ .

 - The first layer of the network (the input layer) is to splice the first and last of $C(w_{t-n+1}), ... , C(w_{t-2}), C(w_{t-1})$ these n-1 vectors to form a vector of size $(n-1) * m $, denoted as $x$.

 - The second layer of the network (the hidden layer) is just like a normal neural network, which uses a fully connected layer, and then passes through the fully connected layer and is processed by the $tanh$ activation function.
 - The third layer of the network (the output layer) has a total of $V$ nodes ( $V$ stands for the vocabulary of the corpus), and essentially this output layer is also a fully connected layer. Each output node $y_i$ represents the un-normalized log probability of the next word $i$. The output value $y$ is finally normalized using a softmax activation function. The maximum probability value is obtained, which is the result we need to predict.

Neural network characteristics:
 - Advantages: Using neural networks to model the constraints between the probability of the current word occurrence and its previous n-1 words, it is clear that this approach has better generalization ability compared to n-gram, as long as the word representation is good enough. This reduces the problem of data sparsity to a large extent.

 - Disadvantages: the modeling ability for long sequences is limited, there may be problems such as long-distance forgetting and gradient disappearance during training, and it is difficult to construct a model with stable long text output.

### 2.3 Pre-trained language modeling based on Transformer
![transformer](/img/12.png)
The Transformer model consists of a number of encoder and decoder layers with strong ability to learn complex semantic information, and many mainstream pre-training models choose the Transformer structure when extracting features, and a series of Transformer-based pre-training models have been produced, including GPT, BERT, T5, etc. These models are able to learn a large number of linguistic representations from a large amount of generalized textual data and apply this knowledge to downstream tasks, obtaining better results.

Pre-training language models are used in a variety of ways:
 - Pre-training: pre-training refers to building a basic model and training it on some more basic datasets and corpora first, and then training it according to specific tasks to learn the general features of the data.
- Fine-tuning: Fine-tuning refers to using the pre-trained model for transfer learning in specific downstream tasks to obtain better generalization effects.

Characteristics of pre-trained language models:
- Advantages: more powerful generalization ability, rich semantic representation, can effectively prevent overfitting.

- Disadvantages: high computational resource requirements, poor interpretability, etc.

### 2.4 Large language model
With the research on pre-trained language models, it has gradually been discovered that there may be a Scaling Law, i.e., as the parameters of a pre-trained model increase exponentially, the performance of its language model also rises linearly. In 2020, OpenAI released GPT-3, which has a reference count of up to 175 billion, and demonstrated for the first time the performance of a large language model.

Compared to previous pre-trained language models with smaller parameter counts, e.g., Bert-large with 330 million parameters and GPT-2 with 1.7 billion parameters, GPT-3 demonstrates a leap in Few-shot language task capability and possesses some capabilities that pre-trained language models do not have. This phenomenon is subsequently referred to as capability emergence. For example, GPT-3 is able to perform contextual learning, completing subsequent tasks based solely on user-given task examples without adjusting the weights. This leap in capability triggered a boom in research on large language models, with major tech giants launching language models with huge number of parameters, such as Meta's LLaMA model with 130 billion parameters and Google's PaLM with 540 billion parameters.

Characteristics of a large language model:
 - Advantages: Intelligent as “human”, with the ability to communicate and chat with humans, and even with the ability to use plug-ins for automated information retrieval.
- Disadvantages: large number of parameters, high arithmetic requirements, generation of partially harmful, biased content, etc.
# LLM

## Background knowledge

### 1. Large Language Model (LLM) Background

The Large Language Model (LLM) is an artificial intelligence model designed to understand and generate human language. Large Language Model
LLMs can handle a wide range of natural language tasks, such as text categorization, question and answer, translation, conversation, etc.

![Elovutionary Tree](/1.png)

In general, a large language model (LLM) is a language model that contains hundreds of billions (or more) of parameters (currently, models with more than 10B parameters are defined as large language models), which are trained on large amounts of textual data, such as the models GPT-3, ChatGPT, PaLM, BLOOM, and LLaMA.
These parameters are trained on large amounts of textual data, such as models GPT-3, ChatGPT, PaLM, BLOOM, and LLaMA.

As of 2023, language model development has gone through three phases:

Phase 1 : Designing a series of self-supervised training objectives (MLM, NSP, etc.), designing novel model architectures (Transformer), following the
Pre-training and Fine-tuning paradigms. Typical representatives are BERT, GPT, XLNet, etc;

Phase 2 : Gradually expand the model parameters and training corpus size, and explore different types of architectures. Typical representatives are BART, T5, GPT-3 and so on;

Phase 3 : Towards the era of AIGC (Artificial Intelligent Generated Content), model parameter scale steps into trillions of dollars, model architecture
for autoregressive architecture, large models towards the era of conversational, generative, multimodal, more focus on aligning with human interaction to achieve reliable, safe,
non-toxic models. Typical representatives are InstructionGPT, ChatGPT, Bard, GPT-4 and so on.

### 2. Language Model (LM)
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

#### 2.1 Rule-based and statistical language modeling (N-gram)
Modeling and analysis of fixed-length text window sequences by manually designing features and using statistical methods is also known as N-gram language modeling. In the example above, the probability of a sentence sequence is calculated using the chain rule, which has two drawbacks:

 - The parameter space is too large: there are too many possibilities $P(W_n|W_1,W_2,...,W_n)$ for conditional probabilities to be estimated, and they are not always useful.

 - The data is sparse: there are many combinations of word pairs that do not appear in the corpus, and the probability based on the maximum likelihood estimation is zero.

In order to solve the above problem, Markov's hypothesis is introduced: the probability of an occurrence of an arbitrary word is related to only a finite number of words or words that occur before it.
 - If the occurrence of a word is independent of the words around it, then we call it a unigram, or a unitary language model.
 
![4](/4.png)

 - If the occurrence of a word depends only on the occurrence of a word before it, then we call it bigram.
 $$ P(S) = P(W_1) * P(W_2|W_1) * P(W_3|W_2)*... * P(W_n|W_{n-1})$$
 - If the occurrence of a word depends only on the two words appearing before it, then we call it trigram.
 $$ P(S) = P(W_1) * P(W_2|W_1) * P(W_3|W_2, W_1)*... * P(W_n|W_{n-1}, W_{n-2})$$
 - In general, the N-tuple model is the assumption that the probability of occurrence of the current word is only related to the N-1 words preceding it, and these probability parameters are all computable from large-scale corpora, such as the ternary probability:
 $$ P(W_i|W_{i-1}, W_{i-2}) = Count(W_{i-2}W_{i-1}W_i) / Count(W_{i-2}W_{i-1})$$
The most used in practice are bigram and trigram, and the next example is the bigram language model to understand how it works:
 - First we prepare a corpus (simply understood as a dataset that allows the model to learn), in order to calculate the parameters of the corresponding binary model, i.e. $P(W_i|W_{i-1})$, we have to count, i.e. $C(W_{i-1},W_i)$, then count $C(W_{i-1})$, and then divide can be used to get the probability.
 - $C(W_{i-1},W_i)$ The counting results are as follows：
 ![count1](/2.png)
 - $C(W_{i-1})$ The counting results are as follows：
 ![count2](/3.png)
 - So how does the bigram language model realize the results of parameter calculation for the above corpus? Suppose, I want to calculate think $P(_想|_我)≈ 0.38$ , the calculation process is shown as follows: (other parameters calculation process is similar)
 $$P(_想|_我)≈ {\frac{C(_我,_想)}{C(_我)}} = {\frac{800}{2100}} ≈0.38 $$
 - If the binary model (bigram) for this corpus is built, our target computation can be realized.
 - Calculate the probability of a sentence, as an example:
 $$P(_{我想去打篮球})=P(_想|_我) * P(_去|_想) * P(_打|_去) * P(_{篮球}|_打)= {\frac{800}{2100} * \frac{600}{900} * \frac{690}{2000} * \frac{20}{800}} ≈ 0.0022 $$ 
 - Predict the most likely next word in a sentence, e.g., 我想去打 [mask]? Think: mask = 篮球 or mask = 晚饭.
$$P(_{我想去打篮球}) =≈ 0.0022 $$
$$P(_{我想去打晚饭}) =≈ 0.00022 $$
 - It can be seen that $P(_{我想去打篮球}) > P(_{我想去打晚饭})$, so mask = 篮球, contrasting with the real context and also with human habits. 

Characteristics of the N-gram language model:
 - Advantages: using great likelihood estimation, parameters are easy to train; completely contains all the information of the first n-1 words; highly interpretable, intuitive and easy to understand.
 - Disadvantages: lack of long-term, can only model the first n-1 words; as n increases, the parameter space grows exponentially; sparse data, inevitably OOV
problems; based solely on statistical frequency, poor generalization ability.

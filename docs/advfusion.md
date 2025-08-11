# Advfusion

This document details Advfusion and its changes compared to the previous version. For details on the core idea refer to the [original paper](https://arxiv.org/pdf/2307.07854).

## Current Version Changes

The main difference from the previous study is the exclusion of encoder-only and encoder-decoder models from the experiments.
As a result:

- The distinction between language adapters and task adapters presented in the original paper is considerably reduced.

- All training is done using _Causal Language Modeling (CLM)_ which enables next token prediction, with _Masked Language Modeling (MLM)_ no longer employed.

> [!NOTE]
> On **training objectives**: MLM and text generation are conducted using different objectives depending on the model type.
>
> - encoder-only and encoder-decoder:
>     - MLM: traditional MLM for encoder-only and Span Corruption / Masked Span Modeling for encoder-decoder models.
>     - Text generation: sequence-to-sequence modeling for encoder-decoder (and encoder-only by stacking a decoder).
> - decoder-only:
>     - MLM: typically not done for decoder-only since they do not have bidirectional attention.
>     - Text generation: next token prediction using CLM.

### Language vs Task Adapters

In the original paper, Language and task adapters had 2 main differences.

> [!NOTE]
> The original paper does not train task adapters. The following is its definition of Task Adapters.
> However, the adversarial fusion adapter is trained in a fashion very similar to task adapters.

**Training Data**: language adapters are trained on textual data in a specific programming language (PL).
The data typically is not in a task or instruction format.
task adapters however, are trained on data that is formatted in a way to satisfy the target task.

**Training Objective**: language adapters are trained using MLM in order to learn language-specific context.
On the other hand, task adapters (and the adversarial fusion adapter) are trained on text generation.

In this study, we only include more recent decoder-only models.
That means **_the objective for all adapters will be next token generation using CLM_**.
This reduces the distinction between language adapters and task adapters to their training data.

Therefore, a language adapter is an adapter trained on unformatted data pertaining to a single programming language (with no natural language instructions).
A task adapter on the other hand, is trained on formatted data including natural language instructions and programming language.

| Study    | Module             | Training Objective | Training Data |
| -------- | ------------------ | ------------------ | ------------- |
| Original | language adapters  | MLM                | PL Only       |
| Original | Adversarial Fusion | s-t-s              | NL & PL       |
| Current  | language adapters  | CLM                | NL & PL       |
| Current  | Adversarial Fusion | CLM                | NL & PL       |

> [!NOTE]
> We can reduce this distinction even further by training both Language and task adapters on formatted data.
> The distinction would then be applicable among adapters within an entire adversarial fusion training process.
> If the difference between the adapters of an adversarial fusion is the language of their training data (e.g., Python code summarization and Rust code summarization), we categorize them as language adapters.
> However, If the adapters of the adversarial fusion are trained on data with same language but different tasks (e.g., Python code summarization and Python code generation), we categorize them as task adapters.
> The rationale behind this is that recent models are extensively trained on most programming languages, and further fine-tuning adapters on pure programming language data may not provide the models with any additional knowledge.

## Next Token Prediction

Next token prediction is the process of predicting the next token in a sequence given the previous tokens.
We do this using the CLM objective; the model is trained assign probabilities to all tokens in the vocabulary for the next token in the sequence.
This is done by calculating the loss between the predicted token probabilities and the actual next token in the sequence.

Since we are using decoder-only models, we have to use CLM to train all adapter modules (Language, Task, and Adversarial Fusion).
Since the training objective is the same for all adapters, the training data determines the type of adapter.

For **language adapters**, the difference in the training data for a group of adapters used in a adversarial fusion training process is solely the language.

For **task adapters**, the difference in the training data for the adapters is how the data is formatted for the task.
This means that the language is generally the same across the adapters, but the data is formatted differently to suit different tasks.
For example, one adapter may be trained on Python code summarization (Python code, natural language summaries) and another on Python code generation (natural language instructions and Python code).

Finally, the **adversarial fusion adapter** is trained on data pertaining to a specific language and task.

### Examples

The following is a simple example for a code translation task from Go to Lua.

For this task, the input (or the _prompt_) is a Python code snippet along with natural language instructions wrapping the snippet.

```
### Code written in Python:
original = 'Mary had a %s lamb.'
extra = 'little'
original % extra

### Code written in Lua:
```

The target output is then the translation of the code snippet to Lua.

```
str = string.gsub( "Mary had a X lamb.", "X", "little" )
print( str )
```

For the CLM training, the adapter is trained on the concatenation of the input and target output, and will predict all tokens in the resulting sequence:

```
### Code written in Python:
original = 'Mary had a %s lamb.'
extra = 'little'
original % extra

### Code written in Lua:
str = string.gsub( "Mary had a X lamb.", "X", "little" )
print( str )
```

We may also choose to train the adapter solely on programming language data (For language adapters). For this task for example, we can train unformatted Lua code snippets.
This way, the model learns Lua code syntax and semantics, but does not learn the task's formatting.

> [!NOTE]
> Training adapters on unformatted programming language data may not provide the model with any additional knowledge.
> This is because recent models are extensively trained on most programming languages, and further fine-tuning adapters on pure programming language data may not provide the models with any additional knowledge.

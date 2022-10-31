# bengal

Easy-to-use Go implementations of two Naive Bayes classifiers for multilabel text classification.

## install

In your project:

`$ go get -u github.com/haydenhigg/bengal/v2`

Then, import it as:
```go
import "github.com/haydenhigg/bengal/v2"
```

## use

### modeling

- `TrainMultinomial(input, output [][]string) NaiveBayesModel`: Creates a multinomial model from tokenized inputs.
- `(model *NaiveBayesModel) PredictMultinomial(x []string) []string`: Predicts the labels of a tokenized input.
- `TrainBernoulli(input, output [][]string) NaiveBayesModel`: Creates a Bernoulli model from tokenized inputs.
- `(model *NaiveBayesModel) PredictBernoulli(x []string) []string`: Predicts the labels of a tokenized input.

### example

```go
package main

import (
  "fmt"
  "bengal"
)

func main() {
  inputs := [][]string{
    []string{"...", ...},
    []string{"...", ...},
    ...,
  }

  outputs := [][]string{
    []string{"label1-1", "label2-1"},
    []string{"label1-2", "label2-2"},
    ...,
  }
  
  model := bengal.NewBernoulli(inputs, outputs)
  
  fmt.Println(model.PredictBernoulli([]string {"...", ...}))
}
```

## notes

- It is recommended to stem all input examples using something like [this](https://github.com/dchest/stemmer) before training or predicting.

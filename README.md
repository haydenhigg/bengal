# bengal
Optimized Go implementation of Naive Bayes classifiers for multilabel text classification.

## install
In your project:

`$ go get github.com/haydenhigg/bengal`

Then, import it as:
```go
import "github.com/haydenhigg/bengal"
```

## use
### modeling
- `TrainMultinomial(xs, ys [][]string, smoothing float64) NaiveBayesModel`: Creates and trains a multinomial model.
- `(model *NaiveBayesModel) PredictMultinomial(x []string) []string`: Predicts the labels for an input using token presence only.
- `NewBernoulli(xs, ys [][]string, smoothing float64) NaiveBayesModel`: Creates and trains a Bernoulli model.
- `(model *NaiveBayesModel) PredictBernoulli(x []string) []string`: Predicts the labels for an input using token presence and absence.

### example
```go
package main

import (
  "fmt"
  "bengal"
)

func main() {
	inputs := [][]string{
		[]string{"the", "cat", "was", "crying"},
		[]string{"dogs", "like", "to", "smile"},
		...,
	}

	outputs := [][]string{
		[]string{"cat", "sad"},
		[]string{"dog", "happy"},
		...,
	}

	smoothing := 1.0 // fix the zero-probability problem, 1.0 is common
	model := bengal.NewBernoulli(inputs, outputs, smoothing)

	fmt.Println(model.PredictBernoulli([]string{...}))
}
```

## notes
- It is recommended to stem all input examples using something like [this](https://github.com/dchest/stemmer) before training or predicting.
- This uses log probabilities and smoothing for robustness.
- It's possible to use a prediction function that does not match the training function. i.e. to use `NewBernoulli` for training but `PredictMultinomial` for faster -- and similarly accurate on most data -- predictions.

# bengal

Easy-to-use Go implementation of the multinomial Naive Bayes for multilabel text classification.

## install

In the terminal:

`go get github.com/haydenhigg/bengal`

Then, import it as:
```go
import "github.com/haydenhigg/bengal"
```

## use

### modeling

- `NewModel(input, output [][]string) MultinomialNB`: Creates a model from tokenized inputs.
- `(model *MultinomialNB) Predict(x []string) []string`: Predicts the classes of a tokenized input.

### example

```go
package main

import (
  "fmt"
  "bengal"
)

func main() {
  inputs := [][]string {[]string {"...", ...}, []string {"...", ...}, ...}
  outputs := [][]string {[]string {"label1-1", "label2-1"}, []string {"label1-2", "label2-2"}, ...}
  
  model := bengal.NewModel(inputs, outputs)
  
  fmt.Println(model.Predict([]string {"...", ...}))
}
```

## notes

- It is recommended to stem all input examples using something like [this](https://github.com/dchest/stemmer) before training or predicting.

# bengal

Easy-to-use Go implementation of the multinomial Naive Bayes for multilabel text classification.

## install

`$ go get -u github.com/haydenhigg/bengal`

Then, import it as:
```go
import "github.com/haydenhigg/bengal"
```

## use

### modeling

- `NewModelFromVectors(input, output [][]string) MultinomialNB`: Creates a model from tokenized inputs.
- `(model *MultinomialNB) PredictVector(x []string) []string`: Predicts the classes of the tokenized input.

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

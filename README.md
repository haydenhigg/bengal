# bengal

Easy-to-use Multinomial Naive Bayes for text classification, optimized for multiple output features.

## install

Install the Porter2 implementation with:
`$ go get -u github.com/dchest/stemmer/porter2`

Then, while in your project directory:
`$ git clone https://github.com/haydenhigg/bengal`

Finally, import it as:
```go
import "./bengal"
```

## use

### modelling

- `StemExample(text string) []string`: Tokenizes a string.
- `StemExamples(examples []string) [][]string`: Tokenizes a slice of strings using `StemExample`.
- `NewModel(examples []string, output [][]string) MultinomialNB`: Tokenizes the inputs using `StemExamples` and creates a model from them.
- `NewModelFromVectors(input [][]string, output [][]string) MultinomialNB`: Creates a model from tokenized inputs.
- `(model MultinomialNB) Predict(example string) []string`: Predicts the classes of the input example using `StemExample`.
- `(model MultinomialNB) PredictVector(input []string) []string`: Predicts the classes of the tokenized input.

### serializing

- `(model *MultinomialNB) Serialize() ([]byte, error)`: Serializes a model to a JSON-formatted byte slice.
- `(model *MultinomialNB) SerializeTo(filename string) ([]byte, error)`: Serializes a model to a JSON byte slice and write it to file `filename` (creates the file if it does not exist).
- `Deserialize(bytes []byte) (*MultinomialNB, error)`: Deserializes a model from a JSON-formatted byte slice.
- `DeserializeFrom(filename string) (*MultinomialNB, error)`: Deserializes a model from a JSON-formatted byte slice read from file `filename`.

### example

```go
package main

import (
  "fmt"
  "./bengal"
)

func main() {
  inputs := []string{"...", "...", ...}
  outputs := []string{[]string{"feature1class1", "feature2class1"}, []string{"feature1class2", "feature2class2"}, ...}
  
  model := bengal.NewModel(inputs, outputs)
  
  fmt.Println(model.Predict("..."))
}
```

## clarifications

The multiple output features are (as far as I know) a novel addition to the algorithm, and are only helpful for select tasks. To clarify:
1) **This implementation still functions exactly like a regular Multinomial Naive Bayes classifier if each element in the output slice has only one element.**
2) **The output features are completely independent from one another, as if they were part of two separate classifiers -- for performance and memory efficiency, however, it is easier to clump them together as "features" rather than creating two separate classifiers.

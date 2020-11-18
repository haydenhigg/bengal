package bengal

import (
	"strings"
	"regexp"
	"github.com/dchest/stemmer/porter2" // snowball stemmer
)

func StemSentence(text string) []string {
	unallowed := regexp.MustCompile(`[^0-9a-z ]`)
	stem := porter2.Stemmer.Stem

	words := strings.Fields(strings.ToLower(strings.Trim(text, `\t\n\f\r `)))
	ret := make([]string, len(words))

	for i, w := range words {
		ret[i] = stem(unallowed.ReplaceAllLiteralString(w, ""))
	}
	
	return ret
}

func StemExamples(examples []string) [][]string {
	ret := make([][]string, len(examples))

	for i, ex := range examples {
		ret[i] = StemSentence(ex)
	}

	return ret
}

type MultioutputMultinomialNB struct {
	vocabulary 	[]string							// all unique words in training set
	Classes 	[][]string							// all unique classes per output feature

	Prior 		[]map[string]float64				// class frequencies per output feature
	Condprob 	[]map[string]map[string]float64		// conditional probability maps per output feature

	Input 		[][]string
	Output 		[][]string
}

func NewModelFromVectors(input, output [][]string) MultioutputMultinomialNB {
	features := output[0]

	vocabulary := unique(flatten2d(input))
	classes := make([][]string, len(features))

	prior := make([]map[string]float64, len(features))
	condprob := make([]map[string]map[string]float64, len(features))

	for i := range features {
		classes[i] = unique(getColumn(output, i))

		prior[i] = make(map[string]float64)
		condprob[i] = make(map[string]map[string]float64)

		for _, c := range classes[i] {
			var examplesInClass [][]string
			for j, inp := range input {
				if output[j][i] == c {
					examplesInClass = append(examplesInClass, inp)
				}
			}

			prior[i][c] = float64(len(examplesInClass)) / float64(len(input))

			tokenCountsForClass := make([]int, len(vocabulary))
			for j, token := range vocabulary {
				tc := 0
				for _, word := range flatten2d(examplesInClass) { if word == token { tc++ } }

				tokenCountsForClass[j] = tc
			}

			for t, token := range vocabulary {
				if _, ok := condprob[i][token]; !ok {
					condprob[i][token] = make(map[string]float64)
				}

				condprob[i][token][c] = tryDivide(tokenCountsForClass[t], sum(tokenCountsForClass))
			}
		}
	}

	return MultioutputMultinomialNB{
		vocabulary: vocabulary,
		Classes: classes,

		Prior: prior,
		Condprob: condprob,
		
		Input: input,
		Output: output}
}

func NewModel(examples []string, output [][]string) MultioutputMultinomialNB {
	return NewModelFromVectors(StemExamples(exampleInput), output)
}

func (model MultioutputMultinomialNB) PredictVector(input []string) []string {
	ret := make([]string, len(model.Classes))

	for i, class := range model.Classes {
		scores := make(map[string]float64)

		for _, c := range class {
			scores[c] = model.Prior[i][c]

			for _, token := range input {
				if cp, ok := model.Condprob[i][token]; ok && cp[c] > 0 {
					scores[c] += cp[c]
				}
			}
		}

		ret[i] = argmax(scores)
	}

	return ret
}

func (model MultioutputMultinomialNB) Predict(example string) []string {
	return model.RawPredict(StemSentence(example))
}
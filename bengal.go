package bengal

import "math"

type MultinomialNB struct {
	vocabulary []string                        // all unique words in training set
	Classes    [][]string                      // all unique classes per output feature

	Prior      []map[string]float64            // class frequencies per output feature
	CondProb   []map[string]map[string]float64 // conditional probability maps per output feature

	Input      [][]string
	Output     [][]string
}

func NewModel(input, output [][]string) MultinomialNB {
	features := len(output[0])

	vocabulary := unique(flatten2d(input))
	classes := make([][]string, features)

	prior := make([]map[string]float64, features)
	condprob := make([]map[string]map[string]float64, features)

	for f := 0; f < features; f++ {
		// Get classes for this feature
		featureClasses := make([]string, len(output))

		for i, y := range output {
			featureClasses[i] = y[f]
		}

		classes[f] = unique(featureClasses)
		
		// Get prior and condprob for this feature
		prior[f] = make(map[string]float64)
		condprob[f] = make(map[string]map[string]float64)

		for _, class := range classes[f] {
			// Find class examples
			var examplesInClass [][]string

			for i, x := range input {
				if output[i][f] == class {
					examplesInClass = append(examplesInClass, x)
				}
			}

			// Define prior probabilities from raw class frequency
			prior[f][class] = math.Log1p(float64(len(examplesInClass)) / float64(len(input)))

			// Count tokens in class examples
			examplesInClassVocabulary := flatten2d(examplesInClass)
			tokenCounts := make(map[string]int)

			for _, token := range examplesInClassVocabulary {
				if _, ok := tokenCounts[token]; ok {
					tokenCounts[token]++
				} else {
					tokenCounts[token] = 1
				}
			}

			// Define conditional probabilities
			for _, token := range vocabulary {
				if _, ok := condprob[f][token]; !ok {
					condprob[f][token] = make(map[string]float64)
				}

				thisTokenCount := 0

				if tokenCount, ok := tokenCounts[token]; ok {
					thisTokenCount = tokenCount
				}

				nonMonotonic := float64(1 + thisTokenCount) / float64(len(examplesInClassVocabulary) + len(vocabulary))
				condprob[f][token][class] = math.Log1p(nonMonotonic)
			}
		}
	}

	return MultinomialNB {
		vocabulary: vocabulary,
		Classes: classes,

		Prior: prior,
		CondProb: condprob,
		
		Input: input,
		Output: output,
	}
}

func (model MultinomialNB) Predict(x []string) []string {
	ret := make([]string, len(model.Classes))

	for f, class := range model.Classes {
		scores := make(map[string]float64)

		for _, class := range class {
			scores[class] = model.Prior[f][class]

			for _, token := range x {
				if condprob, ok := model.CondProb[f][token]; ok {
					scores[class] += condprob[class]
				}
			}
		}

		ret[f] = argmax(scores)
	}

	return ret
}

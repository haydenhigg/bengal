package bengal

import "math"

func TrainMultinomial(input, output [][]string) NaiveBayesModel {
	features := len(output[0])

	vocabulary := unique(flatten2d(input))
	classes := make([][]string, features)

	prior := make([]map[string]float64, features)
	condprob := make([]map[string]map[string]float64, features)

	n := len(input)
	nVocabulary := len(vocabulary)

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
			nClass := len(examplesInClass)
			prior[f][class] = math.Log1p(float64(nClass) / float64(n))

			// Count tokens in class examples
			examplesInClassVocabulary := flatten2d(examplesInClass)
			nExampleVocabulary := len(examplesInClassVocabulary)
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

				nonMonotonic := float64(1 + thisTokenCount) / float64(nExampleVocabulary + nVocabulary)
				condprob[f][token][class] = math.Log1p(nonMonotonic)
			}
		}
	}

	return NaiveBayesModel{
		vocabulary: vocabulary,
		Classes: classes,

		Prior: prior,
		CondProb: condprob,
		
		Input: input,
		Output: output,
	}
}

func (model NaiveBayesModel) PredictMultinomial(x []string) []string {
	ret := make([]string, len(model.Classes))

	for f, feature := range model.Classes {
		scores := make(map[string]float64)

		for _, class := range feature {
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

package bengal

import "math"

func TrainBernoulli(input, output [][]string) NaiveBayesModel {
	features := len(output[0])

	vocabulary := unique(flatten2d(input))
	classes := make([][]string, features)

	prior := make([]map[string]float64, features)
	condprob := make([]map[string]map[string]float64, features)

	n := len(input)

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

			// Count number of examples with each token
			exampleCountsForTokens := make(map[string]int)

			for _, example := range examplesInClass {
				for _, token := range example {
					if _, ok := exampleCountsForTokens[token]; ok {
						exampleCountsForTokens[token]++
					} else {
						exampleCountsForTokens[token] = 1
					}
				}
			}

			// Define conditional probabilities
			for _, token := range vocabulary {
				if _, ok := condprob[f][token]; !ok {
					condprob[f][token] = make(map[string]float64)
				}

				condprob[f][token][class] = float64(1 + exampleCountsForTokens[token]) / float64(2 + nClass)
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

func (model NaiveBayesModel) PredictBernoulli(x []string) []string {
	ret := make([]string, len(model.Classes))

	for f, feature := range model.Classes {
		scores := make(map[string]float64)

		for _, class := range feature {
			scores[class] = model.Prior[f][class]

			for _, token := range x {
				if condprob, ok := model.CondProb[f][token]; ok {
					scores[class] += math.Log1p(condprob[class])
				} else {
					scores[class] += math.Log1p(1 - condprob[class])
				}
			}
		}

		ret[f] = argmax(scores)
	}

	return ret
}

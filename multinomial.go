package bengal

import "math"

func NewMultinomial(xs, ys [][]string, alpha float64) (*Model, error) {
	// confirm assumptions
	n := len(xs)
	if len(ys) != n {
		return nil, MismatchedXsYsError
	}

	numLabels := len(ys[0])
	for _, y := range ys {
		if len(y) != numLabels {
			return nil, YsShapeError
		}
	}

	if alpha <= 0 {
		return nil, AlphaDomainError
	}

	// create model
	model := &Model{
		xs:         xs,
		ys:         ys,
		Vocabulary: deduplicate(flatten2d(xs)),
		Classes:    make([][]string, numLabels),
		Prior:      make([]map[string]float64, numLabels),
		CondProb:   make([]map[string]map[string]float64, numLabels),
	}

	for l := range numLabels {
		// Get classes for this feature
		featureClasses := make([]string, n)
		for i, y := range ys {
			featureClasses[i] = y[l]
		}

		model.Classes[l] = deduplicate(featureClasses)

		// Get prior and condprob for this feature
		model.Prior[l] = make(map[string]float64, len(model.Classes[l]))
		model.CondProb[l] = make(map[string]map[string]float64)

		for _, class := range model.Classes[l] {
			// find class examples
			var xsOfClass [][]string
			for i, x := range xs {
				if ys[i][l] == class {
					xsOfClass = append(xsOfClass, x)
				}
			}

			numXsOfClass := float64(len(xsOfClass))

			// calculate prior probability
			model.Prior[l][class] = math.Log(numXsOfClass / float64(n))

			// calculate conditional probabilities
			numInClassPerToken := make(map[string]float64)
			for _, x := range xsOfClass {
				for _, token := range x {
					if _, ok := numInClassPerToken[token]; !ok {
						numInClassPerToken[token] = 1
					} else {
						numInClassPerToken[token]++
					}
				}
			}

			numTokensInClass := 0.
			for _, v := range numInClassPerToken {
				numTokensInClass += v
			}

			for _, token := range model.Vocabulary {
				if _, ok := model.CondProb[l][token]; !ok {
					model.CondProb[l][token] = make(map[string]float64)
				}

				model.CondProb[l][token][class] = (numInClassPerToken[token] + alpha) / (numTokensInClass + alpha*float64(len(model.Vocabulary)))
			}
		}
	}

	return model, nil
}

func (model *Model) PredictMultinomial(x []string) []string {
	y := make([]string, len(model.Classes))

	// infer the best class per label
	for l, classes := range model.Classes {
		scores := make([]float64, len(classes))

		for c, class := range classes {
			scores[c] = model.Prior[l][class]

			for _, token := range x {
				if tokenCondProb, ok := model.CondProb[l][token]; ok {
					scores[c] += math.Log(tokenCondProb[class])
				}
			}
		}

		y[l] = argmax(classes, scores)
	}

	return y
}

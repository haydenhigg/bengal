package bengal

import (
	"math"
	"slices"
)

func NewBernoulli(xs, ys [][]string, smoothing float64) (*Model, error) {
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

	// create model
	model := &Model{
		xs:         xs,
		ys:         ys,
		Vocabulary: unique(flatten2d(xs)),
		Classes:    make([][]string, numLabels),
		Prior:      make([]map[string]float64, numLabels),
		CondProb:   make([]map[string]map[string]float64, numLabels),
	}

	for l := range numLabels {
		// extract all classes
		featureClasses := make([]string, n)
		for i, y := range ys {
			featureClasses[i] = y[l]
		}

		model.Classes[l] = unique(featureClasses)

		// find probabilities
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
			numXsOfClassPerToken := make(map[string]float64)
			for _, x := range xsOfClass {
				for _, token := range unique(x) {
					numXsOfClassPerToken[token]++
				}
			}

			for _, token := range model.Vocabulary {
				if _, ok := model.CondProb[l][token]; !ok {
					model.CondProb[l][token] = make(map[string]float64)
				}

				model.CondProb[l][token][class] = (numXsOfClassPerToken[token] + smoothing) / (numXsOfClass + 2*smoothing)
			}
		}
	}

	return model, nil
}

func (model *Model) PredictBernoulli(x []string) []string {
	y := make([]string, len(model.Classes))

	// infer the best class per label
	for l, classes := range model.Classes {
		scores := make([]float64, len(classes))

		for c, class := range classes {
			scores[c] = model.Prior[l][class]

			for _, token := range model.Vocabulary {
				if slices.Contains(x, token) {
					scores[c] += math.Log(model.CondProb[l][token][class])
				} else {
					scores[c] += math.Log(1 - model.CondProb[l][token][class])
				}
			}
		}

		y[l] = argmax(classes, scores)
	}

	return y
}

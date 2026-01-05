package bengal

import "errors"

type Model struct {
	xs [][]string
	ys [][]string

	Vocabulary []string
	Classes    [][]string
	Prior      []map[string]float64
	CondProb   []map[string]map[string]float64
}

var MismatchedXsYsError = errors.New("number of inputs does not match number of outputs")
var YsShapeError = errors.New("outputs cannot be aligned")

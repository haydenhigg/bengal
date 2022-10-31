package bengal

type NaiveBayesModel struct{
	vocabulary []string                        // all unique words in training set
	Classes    [][]string                      // all unique classes per output feature

	Prior      []map[string]float64            // class frequencies per output feature
	CondProb   []map[string]map[string]float64 // conditional probability maps per output feature

	Input      [][]string
	Output     [][]string
}

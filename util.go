package bengal

import "math"

func unique(arr []string) []string {
	uniqueTokens := make(map[string]struct{})

	for _, v := range arr {
		if _, ok := uniqueTokens[v]; !ok {
			uniqueTokens[v] = struct{}{}
		}
	}

	keys := make([]string, len(uniqueTokens))
	i := 0

	for token := range uniqueTokens {
		keys[i] = token
		i++
	}

	return keys
}

func flatten2d(mat [][]string) []string {
	var ret []string

	for _, r := range mat {
		ret = append(ret, r...)
	}

	return ret
}

func argmax(m map[string]float64) string {
	bestArg := ""
	bestVal := math.Inf(-1)

	for k, v := range m {
		if v > bestVal {
			bestArg = k
			bestVal = v
		}
	}

	return bestArg
}

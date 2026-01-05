package bengal

import "math"

func unique[T comparable](s []T) []T {
	deduplicated := make(map[T]struct{})

	for _, v := range s {
		if _, ok := deduplicated[v]; !ok {
			deduplicated[v] = struct{}{}
		}
	}

	keys := make([]T, len(deduplicated))
	i := 0

	for v := range deduplicated {
		keys[i] = v
		i++
	}

	return keys
}

func flatten2d[T any](s [][]T) []T {
	var flattened []T

	for _, v := range s {
		flattened = append(flattened, v...)
	}

	return flattened
}

func argmax[T any](keys []T, values []float64) T {
	var bestKey T
	bestValue := math.Inf(-1)

	for i, value := range values {
		if value > bestValue {
			bestKey = keys[i]
			bestValue = value
		}
	}

	return bestKey
}

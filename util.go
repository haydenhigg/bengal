package bengal

import "math"

func mapOf[T comparable](s []T) map[T]struct{} {
	m := make(map[T]struct{})

	for _, v := range s {
		if _, ok := m[v]; !ok {
			m[v] = struct{}{}
		}
	}

	return m
}

func deduplicate[T comparable](s []T) []T {
	m := mapOf(s)
	keys := make([]T, len(m))
	i := 0

	for v := range m {
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

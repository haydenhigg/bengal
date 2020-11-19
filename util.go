/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

package bengal

import "math"

func unique(arr []string) []string {
	uniqueWords := make(map[string]struct{})

	for _, k := range arr {
		if _, ok := uniqueWords[k]; !ok {
			uniqueWords[k] = struct{}{}
		}
	}

	keys := make([]string, len(uniqueWords))
	i := 0

    for k := range uniqueWords {
		keys[i] = k
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

func getColumn(mat [][]string, index int) []string {
	ret := make([]string, len(mat))

	for i, r := range mat {
		ret[i] = r[index]
	}

	return ret
}

func sum(arr []int) int {
	ret := 0

	for _, v := range arr { ret += v }

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

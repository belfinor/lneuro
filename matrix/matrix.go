package matrix

// @author  Mikhail Kirillov <mikkirillov@yandex.ru>
// @version 1.000
// @date    2019-03-28

import (
	"math/rand"
)

func New(rows, colums int, value float64) [][]float64 {
	mat := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		mat[i] = make([]float64, colums)
		for j := 0; j < colums; j++ {
			mat[i][j] = value
		}
	}
	return mat
}

func Random(rows, colums int, lower, upper float64) [][]float64 {
	mat := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		mat[i] = make([]float64, colums)
		for j := 0; j < colums; j++ {
			mat[i][j] = rand.Float64()*(upper-lower) + lower
		}
	}
	return mat
}

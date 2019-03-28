package neuro

// @author  Mikhail Kirillov <mikkirillov@yandex.ru>
// @version 1.001
// @date    2019-03-28

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/belfinor/lneuro/matrix"
)

type Net struct {
	HiddenLayer      []float64   // скрытый слой
	InputLayer       []float64   // входной слой
	OutputLayer      []float64   // выходной слой
	WeightHidden     [][]float64 // матрица весов скрытого слоя
	WeightOutput     [][]float64 // матрица весов для выходного слоя
	ErrOutput        []float64   // ошибка на выходном слое
	ErrHidden        []float64   // ошибка на скрытом слое
	LastChangeHidden [][]float64
	LastChangeOutput [][]float64
	Regression       bool    // если true, то просто сложение иначи сигмоида
	Rate1            float64 // коэффициент для скртого слоя
	Rate2            float64 // коэффциент для выходного слоя
}

// сигмоида
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, -float64(x)))
}

// поризводная от сигмоиды. y - значение сигмоиды
func dsigmoid(y float64) float64 {
	return y * (1.0 - y)
}

func (nn *Net) Save(filename string) error {
	out_f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer out_f.Close()
	encoder := json.NewEncoder(out_f)
	err = encoder.Encode(nn)
	if err != nil {
		return err
	}

	return nil
}

func Load(filename string) (*Net, error) {
	in_f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer in_f.Close()

	decoder := json.NewDecoder(in_f)
	nn := &Net{}
	err = decoder.Decode(nn)
	if err != nil {
		return nil, err
	}

	return nn, nil
}

func Defaultk(iInputCount, iHiddenCount, iOutputCount int, iRegression bool) *Net {
	return New(iInputCount, iHiddenCount, iOutputCount, iRegression, 0.25, 0.1)
}

func New(iInputCount, iHiddenCount, iOutputCount int, iRegression bool, iRate1, iRate2 float64) *Net {

	iInputCount += 1
	iHiddenCount += 1

	rand.Seed(time.Now().UnixNano())

	net := &Net{}

	net.Regression = iRegression

	net.Rate1 = iRate1
	net.Rate2 = iRate2

	net.InputLayer = make([]float64, iInputCount)
	net.HiddenLayer = make([]float64, iHiddenCount)
	net.OutputLayer = make([]float64, iOutputCount)

	net.ErrOutput = make([]float64, iOutputCount)
	net.ErrHidden = make([]float64, iHiddenCount)

	net.WeightHidden = matrix.Random(iHiddenCount, iInputCount, -1.0, 1.0)
	net.WeightOutput = matrix.Random(iOutputCount, iHiddenCount, -1.0, 1.0)

	net.LastChangeHidden = matrix.New(iHiddenCount, iInputCount, 0.0)
	net.LastChangeOutput = matrix.New(iOutputCount, iHiddenCount, 0.0)

	return net
}

func (nn *Net) Forward(input []float64) ([]float64, error) {
	if len(input)+1 != len(nn.InputLayer) {
		return nil, errors.New("amount of input variable doesn't match")
	}
	for i := 0; i < len(input); i++ {
		nn.InputLayer[i] = input[i]
	}
	nn.InputLayer[len(nn.InputLayer)-1] = 1.0 //bias node for input layer

	for i := 0; i < len(nn.HiddenLayer)-1; i++ {
		sum := 0.0
		for j := 0; j < len(nn.InputLayer); j++ {
			sum += nn.InputLayer[j] * nn.WeightHidden[i][j]
		}
		nn.HiddenLayer[i] = sigmoid(sum)
	}

	nn.HiddenLayer[len(nn.HiddenLayer)-1] = 1.0 //bias node for hidden layer
	for i := 0; i < len(nn.OutputLayer); i++ {
		sum := 0.0
		for j := 0; j < len(nn.HiddenLayer); j++ {
			sum += nn.HiddenLayer[j] * nn.WeightOutput[i][j]
		}
		if nn.Regression {
			nn.OutputLayer[i] = sum
		} else {
			nn.OutputLayer[i] = sigmoid(sum)
		}
	}
	return nn.OutputLayer[:], nil
}

func (nn *Net) Feedback(target []float64) {
	for i := 0; i < len(nn.OutputLayer); i++ {
		nn.ErrOutput[i] = nn.OutputLayer[i] - target[i]
	}

	for i := 0; i < len(nn.HiddenLayer)-1; i++ {
		err := 0.0
		for j := 0; j < len(nn.OutputLayer); j++ {
			if nn.Regression {
				err += nn.ErrOutput[j] * nn.WeightOutput[j][i]
			} else {
				err += nn.ErrOutput[j] * nn.WeightOutput[j][i] * dsigmoid(nn.OutputLayer[j])
			}

		}
		nn.ErrHidden[i] = err
	}

	for i := 0; i < len(nn.OutputLayer); i++ {
		for j := 0; j < len(nn.HiddenLayer); j++ {
			change := 0.0
			delta := 0.0
			if nn.Regression {
				delta = nn.ErrOutput[i]
			} else {
				delta = nn.ErrOutput[i] * dsigmoid(nn.OutputLayer[i])
			}
			change = nn.Rate1*delta*nn.HiddenLayer[j] + nn.Rate2*nn.LastChangeOutput[i][j]
			nn.WeightOutput[i][j] -= change
			nn.LastChangeOutput[i][j] = change

		}
	}

	for i := 0; i < len(nn.HiddenLayer)-1; i++ {
		for j := 0; j < len(nn.InputLayer); j++ {
			delta := nn.ErrHidden[i] * dsigmoid(nn.HiddenLayer[i])
			change := nn.Rate1*delta*nn.InputLayer[j] + nn.Rate2*nn.LastChangeHidden[i][j]
			nn.WeightHidden[i][j] -= change
			nn.LastChangeHidden[i][j] = change

		}
	}
}

func (nn *Net) CalcError(target []float64) float64 {
	errSum := 0.0
	for i := 0; i < len(nn.OutputLayer); i++ {
		err := nn.OutputLayer[i] - target[i]
		errSum += 0.5 * err * err
	}
	return errSum
}

func genRandomIdx(N int) []int {
	A := make([]int, N)
	for i := 0; i < N; i++ {
		A[i] = i
	}
	//randomize
	for i := 0; i < N; i++ {
		j := i + int(rand.Float64()*float64(N-i))
		A[i], A[j] = A[j], A[i]
	}
	return A
}

func (nn *Net) Train(inputs [][]float64, targets [][]float64, iteration int) error {
	if len(inputs[0])+1 != len(nn.InputLayer) {
		return errors.New("amount of input variable doesn't match")
	}
	if len(targets[0]) != len(nn.OutputLayer) {
		return errors.New("amount of output variable doesn't match")
	}

	iter_flag := -1
	for i := 0; i < iteration; i++ {
		idx_ary := genRandomIdx(len(inputs))
		cur_err := 0.0
		for j := 0; j < len(inputs); j++ {
			nn.Forward(inputs[idx_ary[j]])
			nn.Feedback(targets[idx_ary[j]])
			cur_err += nn.CalcError(targets[idx_ary[j]])
			if (j+1)%1000 == 0 {
				if iter_flag != i {
					fmt.Println("")
					iter_flag = i
				}
				fmt.Printf("iteration %vth / progress %.2f %% \r", i+1, float64(j)*100/float64(len(inputs)))
			}
		}
		if (iteration >= 10 && (i+1)%(iteration/10) == 0) || iteration < 10 {
			fmt.Printf("\niteration %vth MSE: %.5f", i+1, cur_err/float64(len(inputs)))
		}
	}
	fmt.Println("\ndone.")
	return nil
}

func (nn *Net) TrainMap(inputs []map[int]float64, targets [][]float64, iteration int) error {
	if len(targets[0]) != len(nn.OutputLayer) {
		return errors.New("amount of output variable doesn't match")
	}

	iter_flag := -1
	for i := 0; i < iteration; i++ {
		idx_ary := genRandomIdx(len(inputs))
		cur_err := 0.0
		for j := 0; j < len(inputs); j++ {
			nn.ForwardMap(inputs[idx_ary[j]])
			nn.FeedbackMap(targets[idx_ary[j]], inputs[idx_ary[j]])
			cur_err += nn.CalcError(targets[idx_ary[j]])
			if (j+1)%1000 == 0 {
				if iter_flag != i {
					fmt.Println("")
					iter_flag = i
				}
				fmt.Printf("iteration %vth / progress %.2f %% \r", i+1, float64(j)*100/float64(len(inputs)))
			}
		}
		if (iteration >= 10 && (i+1)%(iteration/10) == 0) || iteration < 10 {
			fmt.Printf("\niteration %vth MSE: %.5f", i+1, cur_err/float64(len(inputs)))
		}
	}
	fmt.Println("\ndone.")
	return nil
}

func (nn *Net) ForwardMap(input map[int]float64) (output []float64) {
	for k, v := range input {
		nn.InputLayer[k] = v
	}
	nn.InputLayer[len(nn.InputLayer)-1] = 1.0 //bias node for input layer

	for i := 0; i < len(nn.HiddenLayer)-1; i++ {
		sum := 0.0
		for j, _ := range input {
			sum += nn.InputLayer[j] * nn.WeightHidden[i][j]
		}
		nn.HiddenLayer[i] = sigmoid(sum)
	}

	nn.HiddenLayer[len(nn.HiddenLayer)-1] = 1.0 //bias node for hidden layer
	for i := 0; i < len(nn.OutputLayer); i++ {
		sum := 0.0
		for j := 0; j < len(nn.HiddenLayer); j++ {
			sum += nn.HiddenLayer[j] * nn.WeightOutput[i][j]
		}
		if nn.Regression {
			nn.OutputLayer[i] = sum
		} else {
			nn.OutputLayer[i] = sigmoid(sum)
		}
	}
	return nn.OutputLayer[:]
}

func (nn *Net) FeedbackMap(target []float64, input map[int]float64) {
	for i := 0; i < len(nn.OutputLayer); i++ {
		nn.ErrOutput[i] = nn.OutputLayer[i] - target[i]
	}

	for i := 0; i < len(nn.HiddenLayer)-1; i++ {
		err := 0.0
		for j := 0; j < len(nn.OutputLayer); j++ {
			if nn.Regression {
				err += nn.ErrOutput[j] * nn.WeightOutput[j][i]
			} else {
				err += nn.ErrOutput[j] * nn.WeightOutput[j][i] * dsigmoid(nn.OutputLayer[j])
			}

		}
		nn.ErrHidden[i] = err
	}

	for i := 0; i < len(nn.OutputLayer); i++ {
		for j := 0; j < len(nn.HiddenLayer); j++ {
			change := 0.0
			delta := 0.0
			if nn.Regression {
				delta = nn.ErrOutput[i]
			} else {
				delta = nn.ErrOutput[i] * dsigmoid(nn.OutputLayer[i])
			}
			change = nn.Rate1*delta*nn.HiddenLayer[j] + nn.Rate2*nn.LastChangeOutput[i][j]
			nn.WeightOutput[i][j] -= change
			nn.LastChangeOutput[i][j] = change

		}
	}

	for i := 0; i < len(nn.HiddenLayer)-1; i++ {
		for j, _ := range input {
			delta := nn.ErrHidden[i] * dsigmoid(nn.HiddenLayer[i])
			change := nn.Rate1*delta*nn.InputLayer[j] + nn.Rate2*nn.LastChangeHidden[i][j]
			nn.WeightHidden[i][j] -= change
			nn.LastChangeHidden[i][j] = change

		}
	}
}

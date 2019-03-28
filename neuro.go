package neuro

// @author  Mikhail Kirillov <mikkirillov@yandex.ru>
// @version 1.000
// @date    2019-03-28

import (
	"encoding/json"
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

func (self *Net) Forward(input []float64) (output []float64) {
	if len(input)+1 != len(self.InputLayer) {
		panic("amount of input variable doesn't match")
	}
	for i := 0; i < len(input); i++ {
		self.InputLayer[i] = input[i]
	}
	self.InputLayer[len(self.InputLayer)-1] = 1.0 //bias node for input layer

	for i := 0; i < len(self.HiddenLayer)-1; i++ {
		sum := 0.0
		for j := 0; j < len(self.InputLayer); j++ {
			sum += self.InputLayer[j] * self.WeightHidden[i][j]
		}
		self.HiddenLayer[i] = sigmoid(sum)
	}

	self.HiddenLayer[len(self.HiddenLayer)-1] = 1.0 //bias node for hidden layer
	for i := 0; i < len(self.OutputLayer); i++ {
		sum := 0.0
		for j := 0; j < len(self.HiddenLayer); j++ {
			sum += self.HiddenLayer[j] * self.WeightOutput[i][j]
		}
		if self.Regression {
			self.OutputLayer[i] = sum
		} else {
			self.OutputLayer[i] = sigmoid(sum)
		}
	}
	return self.OutputLayer[:]
}

func (self *Net) Feedback(target []float64) {
	for i := 0; i < len(self.OutputLayer); i++ {
		self.ErrOutput[i] = self.OutputLayer[i] - target[i]
	}

	for i := 0; i < len(self.HiddenLayer)-1; i++ {
		err := 0.0
		for j := 0; j < len(self.OutputLayer); j++ {
			if self.Regression {
				err += self.ErrOutput[j] * self.WeightOutput[j][i]
			} else {
				err += self.ErrOutput[j] * self.WeightOutput[j][i] * dsigmoid(self.OutputLayer[j])
			}

		}
		self.ErrHidden[i] = err
	}

	for i := 0; i < len(self.OutputLayer); i++ {
		for j := 0; j < len(self.HiddenLayer); j++ {
			change := 0.0
			delta := 0.0
			if self.Regression {
				delta = self.ErrOutput[i]
			} else {
				delta = self.ErrOutput[i] * dsigmoid(self.OutputLayer[i])
			}
			change = self.Rate1*delta*self.HiddenLayer[j] + self.Rate2*self.LastChangeOutput[i][j]
			self.WeightOutput[i][j] -= change
			self.LastChangeOutput[i][j] = change

		}
	}

	for i := 0; i < len(self.HiddenLayer)-1; i++ {
		for j := 0; j < len(self.InputLayer); j++ {
			delta := self.ErrHidden[i] * dsigmoid(self.HiddenLayer[i])
			change := self.Rate1*delta*self.InputLayer[j] + self.Rate2*self.LastChangeHidden[i][j]
			self.WeightHidden[i][j] -= change
			self.LastChangeHidden[i][j] = change

		}
	}
}

func (self *Net) CalcError(target []float64) float64 {
	errSum := 0.0
	for i := 0; i < len(self.OutputLayer); i++ {
		err := self.OutputLayer[i] - target[i]
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

func (self *Net) Train(inputs [][]float64, targets [][]float64, iteration int) {
	if len(inputs[0])+1 != len(self.InputLayer) {
		panic("amount of input variable doesn't match")
	}
	if len(targets[0]) != len(self.OutputLayer) {
		panic("amount of output variable doesn't match")
	}

	iter_flag := -1
	for i := 0; i < iteration; i++ {
		idx_ary := genRandomIdx(len(inputs))
		cur_err := 0.0
		for j := 0; j < len(inputs); j++ {
			self.Forward(inputs[idx_ary[j]])
			self.Feedback(targets[idx_ary[j]])
			cur_err += self.CalcError(targets[idx_ary[j]])
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
}

func (self *Net) TrainMap(inputs []map[int]float64, targets [][]float64, iteration int) {
	if len(targets[0]) != len(self.OutputLayer) {
		panic("amount of output variable doesn't match")
	}

	iter_flag := -1
	for i := 0; i < iteration; i++ {
		idx_ary := genRandomIdx(len(inputs))
		cur_err := 0.0
		for j := 0; j < len(inputs); j++ {
			self.ForwardMap(inputs[idx_ary[j]])
			self.FeedbackMap(targets[idx_ary[j]], inputs[idx_ary[j]])
			cur_err += self.CalcError(targets[idx_ary[j]])
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
}

func (self *Net) ForwardMap(input map[int]float64) (output []float64) {
	for k, v := range input {
		self.InputLayer[k] = v
	}
	self.InputLayer[len(self.InputLayer)-1] = 1.0 //bias node for input layer

	for i := 0; i < len(self.HiddenLayer)-1; i++ {
		sum := 0.0
		for j, _ := range input {
			sum += self.InputLayer[j] * self.WeightHidden[i][j]
		}
		self.HiddenLayer[i] = sigmoid(sum)
	}

	self.HiddenLayer[len(self.HiddenLayer)-1] = 1.0 //bias node for hidden layer
	for i := 0; i < len(self.OutputLayer); i++ {
		sum := 0.0
		for j := 0; j < len(self.HiddenLayer); j++ {
			sum += self.HiddenLayer[j] * self.WeightOutput[i][j]
		}
		if self.Regression {
			self.OutputLayer[i] = sum
		} else {
			self.OutputLayer[i] = sigmoid(sum)
		}
	}
	return self.OutputLayer[:]
}

func (self *Net) FeedbackMap(target []float64, input map[int]float64) {
	for i := 0; i < len(self.OutputLayer); i++ {
		self.ErrOutput[i] = self.OutputLayer[i] - target[i]
	}

	for i := 0; i < len(self.HiddenLayer)-1; i++ {
		err := 0.0
		for j := 0; j < len(self.OutputLayer); j++ {
			if self.Regression {
				err += self.ErrOutput[j] * self.WeightOutput[j][i]
			} else {
				err += self.ErrOutput[j] * self.WeightOutput[j][i] * dsigmoid(self.OutputLayer[j])
			}

		}
		self.ErrHidden[i] = err
	}

	for i := 0; i < len(self.OutputLayer); i++ {
		for j := 0; j < len(self.HiddenLayer); j++ {
			change := 0.0
			delta := 0.0
			if self.Regression {
				delta = self.ErrOutput[i]
			} else {
				delta = self.ErrOutput[i] * dsigmoid(self.OutputLayer[i])
			}
			change = self.Rate1*delta*self.HiddenLayer[j] + self.Rate2*self.LastChangeOutput[i][j]
			self.WeightOutput[i][j] -= change
			self.LastChangeOutput[i][j] = change

		}
	}

	for i := 0; i < len(self.HiddenLayer)-1; i++ {
		for j, _ := range input {
			delta := self.ErrHidden[i] * dsigmoid(self.HiddenLayer[i])
			change := self.Rate1*delta*self.InputLayer[j] + self.Rate2*self.LastChangeHidden[i][j]
			self.WeightHidden[i][j] -= change
			self.LastChangeHidden[i][j] = change

		}
	}
}

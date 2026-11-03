package colorizer

import (
	"errors"
	"unsafe"

	"github.com/oxzjh/cv"
	"github.com/oxzjh/ort"
)

type Colorizer struct {
	engine *ort.Ort
	width  int
	height int
	rows   int
	cols   int
}

func (c *Colorizer) getChannel(width, height int, p *float32) *cv.Mat {
	channel := cv.NewMatWithData(c.rows, c.cols, cv.MatType32FC1, unsafe.Pointer(p))
	resized := channel.Resize(cv.NewSize(width, height), 0, 0, cv.INTER_LINEAR)
	channel.Free()
	return resized
}

func (c *Colorizer) Coloring(m *cv.Mat) (out *cv.Mat, err error) {
	resized := m.Resize(cv.NewSize(c.width, c.height), 0, 0, cv.INTER_LINEAR)
	resizedLab := getLab(resized)
	resized.Free()
	inputBlob := resizedLab.ConvertTo(cv.MatType32FC1)
	resizedLab.Free()
	defer inputBlob.Free()
	var result *ort.Result
	if result, err = c.engine.Infer(inputBlob.Data(), inputBlob.Total()*inputBlob.ElemSize(), ort.TypeFloat); err != nil {
		return
	}
	defer result.Free()
	dims := c.rows * c.cols
	var data []float32
	if data, err = result.GetFloat32(0, dims<<1, false); err != nil {
		return
	}
	width := m.Cols()
	height := m.Rows()
	originLab := getLab(m)
	channelA := c.getChannel(width, height, &data[0])
	channelB := c.getChannel(width, height, &data[dims])
	merged := cv.Merge([]cv.Mat{*originLab, *channelA, *channelB})
	originLab.Free()
	channelA.Free()
	channelB.Free()
	norm := merged.CvtColor(cv.COLOR_Lab2BGR)
	merged.Free()
	out = norm.ConvertToAB(cv.MatType8UC3, 255, 0)
	norm.Free()
	return
}

func (c *Colorizer) ColoringFile(file string) (*cv.Mat, error) {
	m := cv.IMRead(file, cv.FLAG_COLOR)
	defer m.Free()
	if m.Cols() < 4 || m.Rows() < 4 {
		return nil, errors.New("read file failed")
	}
	return c.Coloring(m)
}

func (c *Colorizer) Close() {
	if c.engine != nil {
		c.engine.Close()
		c.engine = nil
	}
}

func New(model string, threads int) (*Colorizer, error) {
	engine, err := ort.New(model, "colorizer", threads)
	if err != nil {
		return nil, err
	}
	if engine.OutputDims[0][1] != 2 {
		engine.Close()
		return nil, errors.New("invalid model")
	}
	return &Colorizer{engine, int(engine.InputDims[0][3]), int(engine.InputDims[0][2]), int(engine.OutputDims[0][2]), int(engine.OutputDims[0][3])}, nil
}

func getLab(m *cv.Mat) *cv.Mat {
	norm := m.ConvertToAB(cv.MatType32FC3, 1/255.0, 0)
	lab := norm.CvtColor(cv.COLOR_BGR2Lab)
	norm.Free()
	splits := lab.Split()
	lab.Free()
	for i := 1; i < len(splits); i++ {
		splits[i].Free()
	}
	return &splits[0]
}

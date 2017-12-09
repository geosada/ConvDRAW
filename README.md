# ConvDRAW

TensorFlow Implementation of DeepMind’s Convolutional DRAW: [“Towards Conceptual Compression”](http://papers.nips.cc/paper/6542-towards-conceptual-compression) (NIPS 2016) on the SVHN generation task.

<div style="text-align: center;">
  <img src="https://raw.githubusercontent.com/geosada/ConvDRAW/img/test.gif" width="512">
</div>

The DRAW process proceeds in 8 steps from the leftmost column to the second column from the left and the rightmost column is original images. 

<div style="text-align: center;">
  <img src="https://raw.githubusercontent.com/geosada/ConvDRAW/img/test.png" width="512">
  <img src="https://raw.githubusercontent.com/geosada/ConvDRAW/img/test3.png" width="512">
</div>

This code was based on [Eric Jang's DRAW](https://github.com/ericjang/draw).

## Usage

```python main.py```

## ToDo

There seems to be something bug in tensor multiplication in write attention function therefore it turns off now.


## Useful Resources

- [Eric Jang's implementaion of DRAW](https://github.com/ericjang/draw)

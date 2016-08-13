# tsnetwork

Graph layouts by t-distributed stochastic neighbour embedding.

This repository contains the implementation of a graph layout algorithm that makes use of the [t-SNE](https://lvdmaaten.github.io/tsne/) dimensionality reduction technique.
The exploration and evaluation of using this technique was done as my MSc thesis project.

## Dependencies

This was developed and tested solely on ArchLinux.
For this implementation to work, you will need:

* Python 3+
* [NumPy](http://www.numpy.org/)
* [graph-tool](https://graph-tool.skewed.de/)
* [Theano](http://deeplearning.net/software/theano/)
* [Graphviz](http://www.graphviz.org/)

For rendering fancy animations (even more heavily untested, probably only works on my system) you need:

* [ffmpeg](https://ffmpeg.org/)
* [ImageMagick](https://www.imagemagick.org/)

## Layout animations

Animations that show the state of the layout as a function of optimization time can be seen [over here](https://hankruiger.github.io/tsnetwork/animations.html).

## Warning

Usage of this software is at your own risk.
This utility writes and removes directories in a directory you specify, and (being __not__ a professional software developer) you should not trust using this if you're afraid to lose data.

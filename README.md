# `tsnetwork`

Graph layouts by t-distributed stochastic neighbour embedding.

This repository contains the implementation of a graph layout algorithm that makes use of the [t-SNE](https://lvdmaaten.github.io/tsne/) dimensionality reduction technique.

The exploration and evaluation of using this technique for graph layouts was done as my MSc thesis project at Rijksuniversiteit Groningen, which I aim to finish in August 2016.

## Dependencies

This was developed and tested solely on ArchLinux.
For this implementation to work, you will need:

* `python3`
* [`numpy`](http://www.numpy.org/)
* [`graph-tool`](https://graph-tool.skewed.de/)
* [`theano`](http://deeplearning.net/software/theano/)
* [`graphviz`](http://www.graphviz.org/)

For rendering fancy animations (even more heavily untested, probably only works on my system) you need:

* [`ffmpeg`](https://ffmpeg.org/)
* [`imagemagick`](https://www.imagemagick.org/)

## Benchmark layout animations

For a set of graphs that has been used as a benchmark, animations that show the state of the layout as a function of optimization time can be seen [over here](https://hankruiger.github.io/tsnetwork/animations).

## Warning

Usage of this software is at your own risk.
This utility writes and removes directories in a directory you specify, and (with me being __not__ a professional software developer) you should not trust using this if you're afraid to lose data.

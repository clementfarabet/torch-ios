# nnx: an Xperimental package for neural network modules + optimizations

The original neural network from Torch7, 'nn', contains stable and widely
used modules. 'nnx' contains more experimental, unproven modules, and
optimizations. Eventually, modules that become stable enough will make 
their way into 'nn' (some already have).

## Install 

1/ Torch7 is required:

Dependencies, on Linux (Ubuntu > 9.04):

``` sh
$ apt-get install gcc g++ git libreadline5-dev cmake wget libqt4-core libqt4-gui libqt4-dev
```

Dependencies, on Mac OS (Leopard, or more), using [Homebrew](http://mxcl.github.com/homebrew/):

``` sh
$ brew install git readline cmake wget qt
```

Then on both platforms:

``` sh
$ git clone https://github.com/andresy/torch
$ cd torch
$ mkdir build; cd build
$ cmake ..
$ make
$ [sudo] make install
```

2/ Once Torch7 is available, install this package:

``` sh
$ [sudo] torch-pkg install nnx
```

## Use the library

First run torch, and load nnx:

``` sh
$ torch
``` 

``` lua
> require 'nnx'
```

Once loaded, tab-completion will help you navigate through the
library (note that most function are added directly to nn):

``` lua
> nnx. + TAB
...
> nn. + TAB
```

In particular, it's good to verify that all modules provided pass their
tests:

``` lua
> nnx.test_all()
> nnx.test_omp()
```

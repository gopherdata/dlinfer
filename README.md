# A "deep learning" inference engine for Go

This project intends to make trained models, such as AlexNet, GoogleNet, etc., accessible within Go.  It utilizes the functionality of [Intel's inference engine](https://software.intel.com/en-us/deep-learning-sdk) (part of the deep learning SDK) to interact with trained and optimized neural networks. 

_Note_: This is work in progress.

# Dependencies

- [Go 1.8+](https://golang.org/)
- Ubuntu 14.04
- [Intel's Deep Learning SDK Deployment Tools](https://software.intel.com/en-us/deep-learning-sdk)
- [swig 3.0.6+](http://www.swig.org/)

# Use

- get the dlinfer package:

    ```
    go get github.com/gopherds/dlinfer
    ```

- declare the following environmental variable:

    ```
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/deep_learning_sdk_2016.1.0.861/deployment_tools/inference_engine/bin/intel64/lib:/opt/intel/deep_learning_sdk_2016.1.0.861/deployment_tools/inference_engine/lib/intel64
    ```

- build/install your Go progams as you normally would with `go build` and `go install`.  See [here](examples/basic_classification/main.go) for an example.
    ```

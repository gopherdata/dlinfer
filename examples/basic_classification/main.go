package main

import "github.com/gopherds/dlinfer"

const (
	model      = "/CaffeNet.xml"
	pluginPath = "/opt/intel/deep_learning_sdk_2016.1.0.861/deployment_tools/inference_engine/lib/intel64/"
	plugin     = "MKLDNNPlugin"
	labelsFile = "/CaffeNet.labels"
	imageFile  = "/cat.bmp"
)

func main() {

	// Create an inference configurator value.
	pluginPaths := dlinfer.NewStringVector()
	pluginPaths.Add(pluginPath)
	configurator := dlinfer.NewInferenceEngineConfigurator(model, pluginPaths, plugin, labelsFile)

	// Load our image.
	images := dlinfer.NewStringVector()
	images.Add(imageFile)
	configurator.LoadImages(images)

	// Load the model.
	configurator.LoadModel()

	// Infer the content of the image.
	configurator.Infer()

	// Get the top results for our image.
	configurator.GetTopResult(5)
}

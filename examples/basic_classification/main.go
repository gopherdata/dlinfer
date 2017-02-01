package main

import (
	"fmt"

	"github.com/gopherds/dlinfer"
)

func main() {

	// Create an inference configurator value.
	model := ""
	pluginPaths := []string{""}
	plugin := ""
	labelsFile := ""
	configurator := dlinfer.NewInferenceEngineConfigurator(model, pluginPaths, plugin, labelsFile)

	fmt.Println(configurator)
}

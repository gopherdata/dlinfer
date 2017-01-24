// All material is licensed under the Apache License Version 2.0, January 2004
// http://www.apache.org/licenses/LICENSE-2.0

package dlinfer

import "os"

// Configurator inncludes the necessary pieces of an
// Inferenence Engine Configurator, as used in the Intel
// Deep Learning iSDK.
type Configurator struct {
	modelFile  string
	pluginPath string
	pluginName string
	labelFile  string
}

// NewConfigurator creates a new configurator for a particular
// trained model.
func NewConfigurator(modelFile, pluginPath, pluginName, labelFile string) (*Configurator, error) {

	// Validate the model file.
	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		return nil, err
	}

	// Return the configurator.
	return &Configurator{
		modelFile:  modelFile,
		pluginPath: pluginPath,
		pluginName: pluginName,
		labelFile:  labelFile,
	}, nil
}

// LoadImage loads an image as input to an inference.
func LoadImage() {

}

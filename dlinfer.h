/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

/**
 * \brief Declaration of methods and classes for working with Inference Engine API
 * \file InferenceEngineConfigurator.h
 * \example inference_engine_classification_sample/core/InferenceEngineConfigurator.h
 */
#pragma once

#include <string>
#include <vector>
#include <ie_so_loader.h>
#include <ie_cnn_net_reader.h>
#include <inference_engine.hpp>
#include <ie_plugin_ptr.hpp>

/**
 * \class InferenceEngineConfigurator
 * \brief This class communicates with Inference Engine
 */
class InferenceEngineConfigurator {
public:
    /**
     * Constructor of InferenceEngineConfigurator class
     * @param modelFile - the path to model in .xml format
     * @param pluginPath - the path to plugin
     * @param pluginName - the name of plugin for prediction
     * @param labelFile - the path to custom file with labels (Default is empty)
     * @return InferenceEngineConfigurator object
     */
    InferenceEngineConfigurator(const std::string &modelFile, const std::vector<std::string> &pluginPath,
                                const std::string &pluginName, const std::string &labelFile = "");

    /**
     * This method loads image for prediction to blob
     * @param images - the image path for prediction
     */
    void loadImages(const std::vector<std::string> &images);

    /**
     * This method loads image for prediction to blob
     * @param images - the image path for prediction
     */
    void loadImages(const std::string &image);

    /**
     * Method needs to call prediction
     */
    void infer();

    /**
     * Function prints perfomance counts
     * @param stream - output stream
     */
    void printGetPerformanceCounts(std::ostream &stream);

    /**
     * Externally specify meanimage values
     */
    void setISLVC2012MeanScalars();

    /**
     * Method to be called prior to infer
     */
    void loadModel();

private:
    InferenceEngine::CNNNetReader network;
    InferenceEngine::Blob::Ptr _input;
    InferenceEngine::TBlob<float>::Ptr _output;
    InferenceEngine::InferenceEnginePluginPtr _plugin;
    std::vector<std::string> _classes;
    bool imageLoaded = false;
    bool wasInfered = false;

    std::vector<std::string> imageNames;

    static std::string make_plugin_name(const std::string &path, const std::string &input);

    bool readLabels(const std::string &fileName);

    InferenceEngine::InferenceEnginePluginPtr  selectPlugin(const std::vector<std::string> &vector,
                                                            const std::string &basic_string);
};


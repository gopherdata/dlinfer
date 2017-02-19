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
 * \brief Implementations of methods for working with Inference Engine API
 * \file InferenceEngineConfigurator.cpp
 * \example inference_engine_classification_sample/core/InferenceEngineConfigurator.cpp
 */
#include "dlinfer.h"
#include <format_reader_ptr.h>
#include <ie_plugin.hpp>
#include <functional>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <stdarg.h>

using namespace InferenceEngine;

static std::string fileNameNoExt(const std::string &filepath) {
    auto pos = filepath.rfind('.');
    if (pos == std::string::npos) return filepath;
    return filepath.substr(0, pos);
}

// trim from both ends (in place)
inline std::string &trim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
}

static inline std::string stringFormat(const char *msg, ...) {
    va_list va;
    va_start(va, msg);
    char buffer[65536];

    vsnprintf(buffer, sizeof(buffer), msg, va);
    va_end(va);
    return buffer;
}


InferenceEngineConfigurator::InferenceEngineConfigurator(const std::string &modelFile,
                                                         const std::vector<std::string> &pluginPath,
                                                         const std::string &pluginName, const std::string &labelFile)
        : _plugin(selectPlugin(pluginPath, pluginName)) /* connect to plugin */, imageLoaded(false) {
    // Create network reader and load it from file
    network.ReadNetwork(modelFile);
    if (!network.isParseSuccess()) THROW_IE_EXCEPTION << "cannot load a failed Model";
    _plugin->Unload();

    // Get file names for files with weights and labels
    std::string binFileName = fileNameNoExt(modelFile) + ".bin";
    network.ReadWeights(binFileName.c_str());

    std::string labelFileName = fileNameNoExt(modelFile) + ".labels";

    // Change path to labels file if necessary
    if (!labelFile.empty()) {
        labelFileName = labelFile;
    }

    // Try to read labels file
    readLabels(labelFileName);
}

/*
 * Method reads labels file
 * @param fileName - the file path
 * @return true if all success else false
 */
bool InferenceEngineConfigurator::readLabels(const std::string &fileName) {
    _classes.clear();

    std::ifstream inputFile;
    inputFile.open(fileName, std::ios::in);
    if (!inputFile.is_open())
        return false;

    std::string strLine;
    while (std::getline(inputFile, strLine)) {
        trim(strLine);
        _classes.push_back(strLine);
    }

    return true;
}

void InferenceEngineConfigurator::loadImages(const std::string &image) {
    std::vector<std::string> imageVector;
    imageVector.push_back(image);
    loadImages(imageVector);
}

void InferenceEngineConfigurator::loadImages(const std::vector<std::string> &images) {
    InferenceEngine::SizeVector inputDims;
    network.getInputDimentions(inputDims);
    size_t batchSize = inputDims.at(inputDims.size() - 1);
    inputDims.at(inputDims.size() - 1) = 1;

    int inputNetworkSize = std::accumulate(inputDims.begin(), inputDims.end(), 1, std::multiplies<size_t>());

    if (!inputDims.size()) {
        THROW_IE_EXCEPTION << "Error: Incorrect network input dimensions!";
    }

    std::vector<std::shared_ptr<unsigned char>> readImages;

    for (auto i = 0; i < images.size(); i++) {
        FormatReader::ReaderPtr reader(images.at(i).c_str());
        if (reader.get() == nullptr) {
            std::cerr << "[WARNING]: Image " << images.at(i) << " cannot be read!" << std::endl;
            continue;
        }
        if (reader->size() != inputNetworkSize) {
            std::cerr << "[WARNING]: Input sizes mismatch, got " << reader->size() << " bytes, expecting "
                      << inputNetworkSize << std::endl;
            continue;
        }
        readImages.push_back(reader->getData());
        imageNames.push_back(images.at(i));
    }

    if (readImages.size() == 0) {
        THROW_IE_EXCEPTION << "Valid input images were not found!";
    }

    if (batchSize == 1) {
        network.getNetwork().setBatchSize(readImages.size());
    } else {
        if (batchSize > readImages.size()) {
            auto readImagesSize = readImages.size();
            size_t diff = batchSize / readImagesSize;

            for (auto i = 1; i < diff; i++) {
                for (auto j = 0; j < readImagesSize; j++) {
                    imageNames.push_back(imageNames.at(j));
                    readImages.push_back(readImages.at(j));
                }
            }
            if (readImagesSize * diff != batchSize) {
                for (auto j = 0; j < batchSize - readImagesSize * diff; j++) {
                    imageNames.push_back(imageNames.at(j));
                    readImages.push_back(readImages.at(j));
                }
            }
        } else if (batchSize < readImages.size()) {
            while (readImages.size() != batchSize) {
                auto name = imageNames.at(imageNames.size() - 1);
                std::cerr << "[WARNING]: Image " << name << " skipped!" << std::endl;
                imageNames.pop_back();
                readImages.pop_back();
            }
        }
    }

    inputDims = network.getNetwork().getInput()->dims;
    InferenceEngine::SizeVector outputDims = network.getNetwork().getOutput()->dims;

    switch (network.getNetwork().getPrecision()) {
        case Precision::FP32 :
            _input = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(inputDims);
            break;
        case Precision::Q78 :
        case Precision::I16 :
            _input = InferenceEngine::make_shared_blob<short, const InferenceEngine::SizeVector>(inputDims);
            break;
        case Precision::U8 :
            _input = InferenceEngine::make_shared_blob<uint8_t, const InferenceEngine::SizeVector>(inputDims);
            break;
        default:
            THROW_IE_EXCEPTION << "Unsupported network precision: " << network.getNetwork().getPrecision();
    }
    _input->allocate();

    _output = InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(outputDims);
    _output->allocate();

    std::shared_ptr<unsigned char> imagesData;
    size_t imagesSize = readImages.size() * inputNetworkSize;
    imagesData.reset(new unsigned char[imagesSize], std::default_delete<unsigned char[]>());

    for (auto i = 0, k = 0; i < readImages.size(); i++) {
        for (auto j = 0; j < inputNetworkSize; j++, k++) {
            imagesData.get()[k] = readImages.at(i).get()[j];
        }
    }

    readImages.clear();

    InferenceEngine::ConvertImageToInput(imagesData.get(), imagesSize, *_input);

    imageLoaded = true;
}

void InferenceEngineConfigurator::infer() {
    if (!imageLoaded) {
        THROW_IE_EXCEPTION << "Scoring failed! Input data is not loaded!";
    }
    InferenceEngine::ResponseDesc dsc;
    InferenceEngine::StatusCode sts = _plugin->Infer(*_input, *_output, &dsc);

    // Check errors
    if (sts == InferenceEngine::GENERAL_ERROR) {
        THROW_IE_EXCEPTION << "Scoring failed! Critical error: " << dsc.msg;
    } else if (sts == InferenceEngine::NOT_IMPLEMENTED) {
        THROW_IE_EXCEPTION << "Scoring failed! Input data is incorrect and not supported!";
    } else if (sts == InferenceEngine::NETWORK_NOT_LOADED) {
        THROW_IE_EXCEPTION << "Scoring failed! " << dsc.msg;
    }
    wasInfered = true;
}

std::vector<InferenceResults> InferenceEngineConfigurator::getTopResult(unsigned int topCount) {
    if (!wasInfered) {
        THROW_IE_EXCEPTION << "Cannot get top results!";
    }
    std::vector<unsigned> results;
    // Get top N results
    InferenceEngine::TopResults(topCount, *_output, results);

    // Save top N results to vector with InferenceEngineConfigurator::InferenceResults objects
    std::vector<InferenceResults> outputResults;
    size_t batchSize = _output->dims()[1];

    topCount = std::min<unsigned int>(_output->dims()[0], topCount);

    if (batchSize != imageNames.size()) {
        THROW_IE_EXCEPTION << "Batch size is not equal to the number of images!";
    }
    for (size_t i = 0; i < batchSize; i++) {
        InferenceResults imageResult(imageNames.at(i));
        for (size_t j = 0; j < topCount; j++) {
            unsigned result = results[i * topCount + j];
            std::string label =
                    result < _classes.size() ? _classes[result] : stringFormat("label #%d", result);
            imageResult.addResult(
                    {static_cast<int>(result), _output->data()[result + i * (_output->size() / batchSize)], label});
        }
        outputResults.push_back(imageResult);
    }
    return outputResults;
}

void InferenceEngineConfigurator::printGetPerformanceCounts(std::ostream &stream) {
    long long totalTime = 0;
    std::map<std::string, InferenceEngine::InferenceEngineProileInfo> perfomanceMap;
    // Get perfomance counts
    _plugin->GetPerformanceCounts(perfomanceMap, nullptr);
    // Print perfomance counts
    stream << std::endl << "Perfomance counts:" << std::endl << std::endl;
    for (std::map<std::string, InferenceEngine::InferenceEngineProileInfo>::const_iterator it = perfomanceMap.begin();
         it != perfomanceMap.end(); ++it) {
        stream << std::setw(30) << std::left << it->first + ":";
        switch (it->second.status) {
            case InferenceEngine::InferenceEngineProileInfo::EXECUTED:
                stream << std::setw(15) << std::left << "EXECUTED";
                break;
            case InferenceEngine::InferenceEngineProileInfo::NOT_RUN:
                stream << std::setw(15) << std::left << "NOT_RUN";
                break;
            case InferenceEngine::InferenceEngineProileInfo::OPTIMIZED_OUT:
                stream << std::setw(15) << std::left << "OPTIMIZED_OUT";
                break;
        }
        stream << std::setw(20) << std::left << "realTime: " + std::to_string(it->second.realTime_uSec);
        stream << " cpu: " << it->second.cpu_uSec << std::endl;
        if (it->second.realTime_uSec > 0) {
            totalTime += it->second.realTime_uSec;
        }
    }
    stream << std::setw(20) << std::left << "Total time: " + std::to_string(totalTime) << " microseconds" << std::endl;
}

/*
 * Set the path to plugin
 * @param input - plugin name
 * @return Plugin path
 */
std::string InferenceEngineConfigurator::make_plugin_name(const std::string &path, const std::string &input) {
    std::string separator = "/";
    if (path.empty())
        separator = "";
    return path + separator + "lib" + input + ".so";
}


void InferenceEngineConfigurator::setISLVC2012MeanScalars() {
    // TODO: Put mean image from user
    network.getNetwork().setMeanScalars({104.00698793f, 116.66876762f, 122.67891434f});
}

void InferenceEngineConfigurator::loadModel() {
    wasInfered = false;
    InferenceEngine::ResponseDesc dsc;
    // TODO: this need to be handled in smart wrapper over inference engine plugin
    InferenceEngine::StatusCode sts = _plugin->LoadNetwork(network.getNetwork(), &dsc);
    if (sts == InferenceEngine::GENERAL_ERROR) {
        THROW_IE_EXCEPTION << dsc.msg;
    } else if (sts == InferenceEngine::NOT_IMPLEMENTED) {
        THROW_IE_EXCEPTION << "Model cannot be loaded! Plugin is not supported this model!";
    }
}

static std::ostream & operator << (std::ostream & os, const Version *version) {
    os << "\tPlugin version ......... ";
    if (nullptr == version) {
        os << "UNKNOWN";
    } else {
        os << version->apiVersion.major << "." << version->apiVersion.minor;
    }

    os << "\n\tPlugin name ............ ";
    if (nullptr == version || version->description == nullptr) {
        std :: cout << "UNKNOWN";
    } else {
        os << version->description;
    }

    os << "\n\tPlugin build ........... ";
    if (nullptr == version || version->buildNumber == nullptr) {
        std :: cout << "UNKNOWN";
    } else {
        os << version->buildNumber;
    }

    return os;
}

InferenceEnginePluginPtr InferenceEngineConfigurator::selectPlugin(const std::vector<std::string> &pluginDirs,
                                                                   const std::string &name) {
    std::stringstream errs;
    for (auto &pluginPath : pluginDirs) {
        try {
            InferenceEnginePluginPtr plugin(make_plugin_name(pluginPath, name));
            const Version *version;
            plugin->GetVersion(version);
            std::cout << version << std::endl;
            return plugin;
        }
        catch (const std::exception &ex) {
            errs << "cannot load plugin: " << name << " from " << pluginPath << ": " << ex.what() << ", skipping\n";
        }
    }
    std::cerr << errs.str();
    THROW_IE_EXCEPTION << "cannot load plugin: " << name;
}

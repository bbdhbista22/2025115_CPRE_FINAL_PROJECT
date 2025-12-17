#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <numeric>

#include "Config.h"
#include "Model.h"
#include "Types.h"
#include "Utils.h"
#include "layers/Convolutional.h"
#include "layers/Dense.h"
#include "layers/Flatten.h"
#include "layers/Layer.h"
#include "layers/MaxPooling.h"
#include "layers/Softmax.h"
#include "layers/BatchNormalization.h"

#ifdef ZEDBOARD
#include <file_transfer/file_transfer.h>
#endif

namespace ML {

// Batch test metadata structure
struct BatchTestSample {
    int sample_id;
    int validation_index;
    std::string file_path;
    std::string filename;
    int true_label;
    std::string true_class;
    std::string binary_file;
};

// Batch test results structure
struct BatchTestResults {
    int total_samples = 0;

    // Separate accuracy tracking for FP32 and INT8
    int correct_top1_fp32 = 0;
    int correct_top1_int8 = 0;
    int correct_top5_fp32 = 0;
    int correct_top5_int8 = 0;

    std::vector<double> fp32_times;
    std::vector<double> int8_times;

    std::vector<int> true_labels;
    std::vector<int> predicted_labels_fp32;
    std::vector<int> predicted_labels_int8;

    std::vector<double> cosine_similarities;
};

// Build AudioCNN_IRMAS model for musical instrument classification
Model buildAudioCNN_IRMAS(const Path modelPath) {
    Model model;
    logInfo("--- Building AudioCNN_IRMAS Model ---");

    // === Convolutional Block 1 ===
    
    // Layer 0: conv1_1 (5x5x1x32)
    // Input: 128x128x1 mel-spectrogram
    // Output: 124x124x32 (valid padding: 128-5+1=124)
    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {128, 128, 1}},                                        // Input Data
        LayerParams{sizeof(fp32), {124, 124, 32}},                                       // Output Data
        LayerParams{sizeof(fp32), {5, 5, 1, 32}, modelPath / "conv1_1_weights.bin"},   // Weights
        LayerParams{sizeof(fp32), {32}, modelPath / "conv1_1_bias.bin"}                // Bias
    );

    // Layer 1: bn1_1 (BatchNorm after conv1_1)
    model.addLayer<BatchNormalizationLayer>(
        LayerParams{sizeof(fp32), {124, 124, 32}},                                  // Input
        LayerParams{sizeof(fp32), {124, 124, 32}},                                  // Output
        LayerParams{sizeof(fp32), {32}, modelPath / "bn1_1_mean.bin"},            // Mean
        LayerParams{sizeof(fp32), {32}, modelPath / "bn1_1_variance.bin"},        // Variance
        LayerParams{sizeof(fp32), {32}, modelPath / "bn1_1_gamma.bin"},           // Gamma
        LayerParams{sizeof(fp32), {32}, modelPath / "bn1_1_beta.bin"}             // Beta
    );

    // Layer 2: conv1_2 (5x5x32x32)
    // Input: 124x124x32
    // Output: 120x120x32 (124-5+1=120)
    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {124, 124, 32}},
        LayerParams{sizeof(fp32), {120, 120, 32}},
        LayerParams{sizeof(fp32), {5, 5, 32, 32}, modelPath / "conv1_2_weights.bin"},
        LayerParams{sizeof(fp32), {32}, modelPath / "conv1_2_bias.bin"}
    );

    // Layer 3: bn1_2 (BatchNorm after conv1_2)
    model.addLayer<BatchNormalizationLayer>(
        LayerParams{sizeof(fp32), {120, 120, 32}},
        LayerParams{sizeof(fp32), {120, 120, 32}},
        LayerParams{sizeof(fp32), {32}, modelPath / "bn1_2_mean.bin"},
        LayerParams{sizeof(fp32), {32}, modelPath / "bn1_2_variance.bin"},
        LayerParams{sizeof(fp32), {32}, modelPath / "bn1_2_gamma.bin"},
        LayerParams{sizeof(fp32), {32}, modelPath / "bn1_2_beta.bin"}
    );

    // Layer 4: pool1 (2x2 max pooling)
    // Input: 120x120x32
    // Output: 60x60x32
    model.addLayer<MaxPoolingLayer>(
        LayerParams{sizeof(fp32), {120, 120, 32}},
        LayerParams{sizeof(fp32), {60, 60, 32}},
        LayerParams{sizeof(fp32), {2, 2}}
    );

    // === Convolutional Block 2 ===

    // Layer 5: conv2_1 (3x3x32x64)
    // Input: 60x60x32
    // Output: 58x58x64 (60-3+1=58)
    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {60, 60, 32}},
        LayerParams{sizeof(fp32), {58, 58, 64}},
        LayerParams{sizeof(fp32), {3, 3, 32, 64}, modelPath / "conv2_1_weights.bin"},
        LayerParams{sizeof(fp32), {64}, modelPath / "conv2_1_bias.bin"}
    );

    // Layer 6: bn2_1 (BatchNorm after conv2_1)
    model.addLayer<BatchNormalizationLayer>(
        LayerParams{sizeof(fp32), {58, 58, 64}},
        LayerParams{sizeof(fp32), {58, 58, 64}},
        LayerParams{sizeof(fp32), {64}, modelPath / "bn2_1_mean.bin"},
        LayerParams{sizeof(fp32), {64}, modelPath / "bn2_1_variance.bin"},
        LayerParams{sizeof(fp32), {64}, modelPath / "bn2_1_gamma.bin"},
        LayerParams{sizeof(fp32), {64}, modelPath / "bn2_1_beta.bin"}
    );

    // Layer 7: conv2_2 (3x3x64x64)
    // Input: 58x58x64
    // Output: 56x56x64 (58-3+1=56)
    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {58, 58, 64}},
        LayerParams{sizeof(fp32), {56, 56, 64}},
        LayerParams{sizeof(fp32), {3, 3, 64, 64}, modelPath / "conv2_2_weights.bin"},
        LayerParams{sizeof(fp32), {64}, modelPath / "conv2_2_bias.bin"}
    );

    // Layer 8: bn2_2 (BatchNorm after conv2_2)
    model.addLayer<BatchNormalizationLayer>(
        LayerParams{sizeof(fp32), {56, 56, 64}},
        LayerParams{sizeof(fp32), {56, 56, 64}},
        LayerParams{sizeof(fp32), {64}, modelPath / "bn2_2_mean.bin"},
        LayerParams{sizeof(fp32), {64}, modelPath / "bn2_2_variance.bin"},
        LayerParams{sizeof(fp32), {64}, modelPath / "bn2_2_gamma.bin"},
        LayerParams{sizeof(fp32), {64}, modelPath / "bn2_2_beta.bin"}
    );

    // Layer 9: pool2 (2x2 max pooling)
    // Input: 56x56x64
    // Output: 28x28x64
    model.addLayer<MaxPoolingLayer>(
        LayerParams{sizeof(fp32), {56, 56, 64}},
        LayerParams{sizeof(fp32), {28, 28, 64}},
        LayerParams{sizeof(fp32), {2, 2}}
    );

    // === Convolutional Block 3 ===

    // Layer 10: conv3_1 (3x3x64x64)
    // Input: 28x28x64
    // Output: 26x26x64 (28-3+1=26)
    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {28, 28, 64}},
        LayerParams{sizeof(fp32), {26, 26, 64}},
        LayerParams{sizeof(fp32), {3, 3, 64, 64}, modelPath / "conv3_1_weights.bin"},
        LayerParams{sizeof(fp32), {64}, modelPath / "conv3_1_bias.bin"}
    );

    // Layer 11: bn3_1 (BatchNorm after conv3_1)
    model.addLayer<BatchNormalizationLayer>(
        LayerParams{sizeof(fp32), {26, 26, 64}},
        LayerParams{sizeof(fp32), {26, 26, 64}},
        LayerParams{sizeof(fp32), {64}, modelPath / "bn3_1_mean.bin"},
        LayerParams{sizeof(fp32), {64}, modelPath / "bn3_1_variance.bin"},
        LayerParams{sizeof(fp32), {64}, modelPath / "bn3_1_gamma.bin"},
        LayerParams{sizeof(fp32), {64}, modelPath / "bn3_1_beta.bin"}
    );

    // Layer 12: conv3_2 (3x3x64x128)
    // Input: 26x26x64
    // Output: 24x24x128 (26-3+1=24)
    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {26, 26, 64}},
        LayerParams{sizeof(fp32), {24, 24, 128}},
        LayerParams{sizeof(fp32), {3, 3, 64, 128}, modelPath / "conv3_2_weights.bin"},
        LayerParams{sizeof(fp32), {128}, modelPath / "conv3_2_bias.bin"}
    );

    // Layer 13: bn3_2 (BatchNorm after conv3_2) - CRITICAL: This fixes the 4% validation failure!
    model.addLayer<BatchNormalizationLayer>(
        LayerParams{sizeof(fp32), {24, 24, 128}},
        LayerParams{sizeof(fp32), {24, 24, 128}},
        LayerParams{sizeof(fp32), {128}, modelPath / "bn3_2_mean.bin"},
        LayerParams{sizeof(fp32), {128}, modelPath / "bn3_2_variance.bin"},
        LayerParams{sizeof(fp32), {128}, modelPath / "bn3_2_gamma.bin"},
        LayerParams{sizeof(fp32), {128}, modelPath / "bn3_2_beta.bin"}
    );

    // Layer 14: pool3 (2x2 max pooling)
    // Input: 24x24x128
    // Output: 12x12x128
    model.addLayer<MaxPoolingLayer>(
        LayerParams{sizeof(fp32), {24, 24, 128}},
        LayerParams{sizeof(fp32), {12, 12, 128}},
        LayerParams{sizeof(fp32), {2, 2}}
    );

    // === Fully Connected Layers ===

    // Layer 15: flatten
    // Input: 12x12x128 = 18,432
    // Output: 18,432
    model.addLayer<FlattenLayer>(
        LayerParams{sizeof(fp32), {12, 12, 128}},
        LayerParams{sizeof(fp32), {18432}}
    );

    // Layer 16: fc1 (Dense 18432 -> 256)
    // Note: ReLU activation is applied in Dense layer
    model.addLayer<DenseLayer>(
        LayerParams{sizeof(fp32), {18432}},
        LayerParams{sizeof(fp32), {256}},
        LayerParams{sizeof(fp32), {18432, 256}, modelPath / "fc1_weights.bin"},
        LayerParams{sizeof(fp32), {256}, modelPath / "fc1_bias.bin"}
    );

    // Layer 17: bn_fc1 (BatchNorm after fc1)
    model.addLayer<BatchNormalizationLayer>(
        LayerParams{sizeof(fp32), {256}},
        LayerParams{sizeof(fp32), {256}},
        LayerParams{sizeof(fp32), {256}, modelPath / "bn_fc1_mean.bin"},
        LayerParams{sizeof(fp32), {256}, modelPath / "bn_fc1_variance.bin"},
        LayerParams{sizeof(fp32), {256}, modelPath / "bn_fc1_gamma.bin"},
        LayerParams{sizeof(fp32), {256}, modelPath / "bn_fc1_beta.bin"}
    );

    // Note: Dropout is skipped during inference

    // Layer 18: fc2 (Dense 256 -> 10 classes)
    // Output: raw logits (no activation yet)
    model.addLayer<DenseLayer>(
        LayerParams{sizeof(fp32), {256}},
        LayerParams{sizeof(fp32), {10}},
        LayerParams{sizeof(fp32), {256, 10}, modelPath / "fc2_weights.bin"},
        LayerParams{sizeof(fp32), {10}, modelPath / "fc2_bias.bin"}
    );

    // Layer 19: softmax (for classification probabilities)
    model.addLayer<SoftmaxLayer>(
        LayerParams{sizeof(fp32), {10}},
        LayerParams{sizeof(fp32), {10}}
    );

    logInfo("AudioCNN_IRMAS Model built successfully!");
    logInfo("Total layers: 20 (8 Conv, 7 BatchNorm, 3 MaxPool, 1 Flatten, 2 Dense, 1 Softmax)");
    
    return model;
}

void runLayerTest(const std::size_t layerNum, const Model& model, const Path& basePath, const LayerData& inputData) {
    logInfo(std::string("--- Running Layer Test ") + std::to_string(layerNum) + " ---");
    
    try {
        Timer timer("Layer Inference");

        // Run inference on the model up to the specified layer
        timer.start();
        
        // Start with layer 0
        model.inferenceLayer(inputData, 0, Layer::InfType::NAIVE);
        const LayerData* output = &model[0].getOutputData();
        
        // Run subsequent layers up to layerNum
        for (std::size_t i = 1; i <= layerNum; i++) {
            model.inferenceLayer(*output, i, Layer::InfType::NAIVE);
            output = &model[i].getOutputData();
        }
        
        timer.stop();
        
        // Debug: Print first few values for problematic layers
        if (layerNum == 6 || layerNum == 7) {
            std::cout << "First 10 output values: ";
            for (size_t i = 0; i < std::min(size_t(10), output->getParams().flat_count()); i++) {
                std::cout << output->get<fp32>(i) << " ";
            }
            std::cout << std::endl;
        }

        // Print the output dimensions
        std::cout << "Layer " << layerNum << " output dimensions: ";
        for (size_t dim : output->getParams().dims) {
            std::cout << dim << " ";
        }
        std::cout << "(total: " << output->getParams().flat_count() << " elements)" << std::endl;

        // Map C++ layer indices to Python feature map filenames
        // Updated architecture: Conv (NO ReLU) → BatchNorm (WITH ReLU)
        // C++ Conv output (no ReLU) → Python Conv output (before BatchNorm)
        // C++ BatchNorm output (with ReLU) → Python ReLU output (after BatchNorm+ReLU)
        const char* pythonLayerMap[] = {
            "layer_0_conv1_1_features.bin",      // C++ 0:  conv1_1 (Conv, NO ReLU)
            "layer_2_relu1_1_features.bin",      // C++ 1:  bn1_1 (BatchNorm+ReLU)
            "layer_3_conv1_2_features.bin",      // C++ 2:  conv1_2 (Conv, NO ReLU)
            "layer_5_relu1_2_features.bin",      // C++ 3:  bn1_2 (BatchNorm+ReLU)
            "layer_6_pool1_features.bin",        // C++ 4:  pool1
            "layer_7_conv2_1_features.bin",      // C++ 5:  conv2_1 (Conv, NO ReLU)
            "layer_9_relu2_1_features.bin",      // C++ 6:  bn2_1 (BatchNorm+ReLU)
            "layer_10_conv2_2_features.bin",     // C++ 7:  conv2_2 (Conv, NO ReLU)
            "layer_12_relu2_2_features.bin",     // C++ 8:  bn2_2 (BatchNorm+ReLU)
            "layer_13_pool2_features.bin",       // C++ 9:  pool2
            "layer_14_conv3_1_features.bin",     // C++ 10: conv3_1 (Conv, NO ReLU)
            "layer_16_relu3_1_features.bin",     // C++ 11: bn3_1 (BatchNorm+ReLU)
            "layer_17_conv3_2_features.bin",     // C++ 12: conv3_2 (Conv, NO ReLU)
            "layer_19_relu3_2_features.bin",     // C++ 13: bn3_2 (BatchNorm+ReLU) - FIXED!
            "layer_20_pool3_features.bin",       // C++ 14: pool3
            "layer_21_flatten_features.bin",     // C++ 15: flatten
            "layer_22_fc1_features.bin",         // C++ 16: fc1 (Dense, NO ReLU)
            "layer_24_relu_fc1_features.bin",    // C++ 17: bn_fc1 (BatchNorm+ReLU)
            "layer_26_fc2_features.bin",         // C++ 18: fc2
            nullptr                              // C++ 19: softmax (not exported)
        };

        // Check if we have a mapping for this layer
        if (layerNum >= 20 || pythonLayerMap[layerNum] == nullptr) {
            std::cout << "No Python feature map available for layer " << layerNum << std::endl;
            std::cout << "Skipping comparison for layer " << layerNum << std::endl;
            return;
        }

        // Load the expected output for this specific layer
        std::string expectedFileName = pythonLayerMap[layerNum];
        Path expectedPath = basePath / expectedFileName.c_str();
        
        // Check if expected file exists
        std::ifstream file(expectedPath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            std::cout << "Expected output file not found: " << expectedPath << std::endl;
            std::cout << "Skipping comparison for layer " << layerNum << std::endl;
            return;
        }
        
        std::streamsize size = file.tellg();
        std::cout << "Expected file size: " << size << " bytes (" << size/4 << " elements)" << std::endl;
        file.close();
        
        // Calculate expected elements
        size_t expectedElements = size / 4;
        size_t outputElements = output->getParams().flat_count();
        
        if (expectedElements != outputElements) {
            std::cout << "DIMENSION MISMATCH: Output has " << outputElements 
                      << " elements, expected " << expectedElements << std::endl;
            return;
        }
        
        // Create LayerData for comparison
        LayerData expected(output->getParams(), expectedPath);
        expected.loadData();
        
        // Debug: Print first few expected values for problematic layers
        if (layerNum == 6 || layerNum == 7) {
            std::cout << "First 10 expected values: ";
            for (size_t i = 0; i < std::min(size_t(10), expected.getParams().flat_count()); i++) {
                std::cout << expected.get<fp32>(i) << " ";
            }
            std::cout << std::endl;
        }
        
        // Compare the outputs
        output->compareWithinPrint<fp32>(expected);
        
    } catch (const std::exception& e) {
        std::cout << "Layer " << layerNum << " test failed: " << e.what() << std::endl;
    }
}

void runInferenceTest(const Model& model, const LayerData& inputData) {
    logInfo("--- Running Full Inference Test ---");

    Timer timer("Full Inference");

    // Run full inference on the model
    timer.start();
    const LayerData& output = model.inference(inputData, Layer::InfType::NAIVE);
    timer.stop();

    // Print output dimensions
    std::cout << "\nFinal output dimensions: ";
    for (size_t dim : output.getParams().dims) {
        std::cout << dim << " ";
    }
    std::cout << "(total: " << output.getParams().flat_count() << " elements)" << std::endl;

    // Print top-5 predictions with instrument names
    const char* instrumentNames[] = {
        "Cello", "Clarinet", "Flute", "Acoustic Guitar", "Electric Guitar",
        "Organ", "Piano", "Saxophone", "Trumpet", "Violin"
    };

    const size_t numClasses = output.getParams().flat_count();
    std::cout << "\nTop-5 predictions:" << std::endl;
    std::vector<std::pair<fp32, size_t>> predictions;
    for (size_t i = 0; i < numClasses; ++i) {
        predictions.push_back({output.get<fp32>(i), i});
    }
    std::sort(predictions.begin(), predictions.end(), std::greater<std::pair<fp32, size_t>>());

    for (size_t i = 0; i < std::min(size_t(5), numClasses); ++i) {
        std::cout << "  " << (i+1) << ". " << instrumentNames[predictions[i].second]
                  << " (class " << predictions[i].second << "): "
                  << (predictions[i].first * 100.0f) << "%" << std::endl;
    }
}

void runQuantizedInferenceTestWithLayerTiming(const Model& model, const LayerData& inputData) {
    logInfo("========================================");
    logInfo("  INT8 QUANTIZED INFERENCE TEST (WITH LAYER TIMING)");
    logInfo("========================================");

    const char* layerNames[] = {
        "conv1_1", "bn1_1", "conv1_2", "bn1_2", "pool1",
        "conv2_1", "bn2_1", "conv2_2", "bn2_2", "pool2",
        "conv3_1", "bn3_1", "conv3_2", "bn3_2", "pool3",
        "flatten", "fc1", "bn_fc1", "fc2", "softmax"
    };

    // First run NAIVE for baseline
    logInfo("\n--- Step 1: Running FP32 NAIVE baseline ---");
    Timer naiveTimer("FP32 NAIVE Inference");
    naiveTimer.start();
    const LayerData& naiveOutput = model.inference(inputData, Layer::InfType::NAIVE);
    naiveTimer.stop();

    // Copy NAIVE output for comparison
    LayerData naiveOutputCopy(naiveOutput);

    // Print NAIVE top prediction
    const char* instrumentNames[] = {
        "Cello", "Clarinet", "Flute", "Acoustic Guitar", "Electric Guitar",
        "Organ", "Piano", "Saxophone", "Trumpet", "Violin"
    };

    size_t naiveTopClass = 0;
    fp32 naiveTopProb = naiveOutput.get<fp32>(0);
    for (size_t i = 1; i < 10; i++) {
        if (naiveOutput.get<fp32>(i) > naiveTopProb) {
            naiveTopProb = naiveOutput.get<fp32>(i);
            naiveTopClass = i;
        }
    }
    std::cout << "FP32 NAIVE Top prediction: " << instrumentNames[naiveTopClass]
              << " (" << (naiveTopProb * 100.0f) << "%)" << std::endl;

    // Now run QUANTIZED with layer-wise timing
    logInfo("\n--- Step 2: Running INT8 QUANTIZED inference with layer timing ---");
    
    std::vector<double> layerTimes;
    layerTimes.resize(model.getNumLayers(), 0.0);
    
    // First layer
    Timer layerTimer("Layer 0");
    layerTimer.start();
    model.inferenceLayer(inputData, 0, Layer::InfType::QUANTIZED);
    layerTimer.stop();
    layerTimes[0] = layerTimer.milliseconds;
    
    // Subsequent layers
    for (size_t i = 1; i < model.getNumLayers(); i++) {
        layerTimer = Timer("Layer " + std::to_string(i));
        layerTimer.start();
        model.inferenceLayer(model[i-1].getOutputData(), i, Layer::InfType::QUANTIZED);
        layerTimer.stop();
        layerTimes[i] = layerTimer.milliseconds;
    }
    
    const LayerData& quantOutput = model.getOutputLayer().getOutputData();

    // Print QUANTIZED top prediction
    size_t quantTopClass = 0;
    fp32 quantTopProb = quantOutput.get<fp32>(0);
    for (size_t i = 1; i < 10; i++) {
        if (quantOutput.get<fp32>(i) > quantTopProb) {
            quantTopProb = quantOutput.get<fp32>(i);
            quantTopClass = i;
        }
    }
    std::cout << "INT8 QUANTIZED Top prediction: " << instrumentNames[quantTopClass]
              << " (" << (quantTopProb * 100.0f) << "%)" << std::endl;

    // Compare outputs
    logInfo("\n--- Step 3: Comparing FP32 vs INT8 results ---");

    // Calculate cosine similarity
    fp32 cosine_sim = naiveOutputCopy.compare<fp32>(quantOutput);
    std::cout << "Cosine similarity: " << (cosine_sim * 100.0f) << "%" << std::endl;

    // Calculate max absolute error
    fp32 max_error = 0.0f;
    for (size_t i = 0; i < 10; i++) {
        fp32 error = std::abs(naiveOutputCopy.get<fp32>(i) - quantOutput.get<fp32>(i));
        max_error = std::max(max_error, error);
    }
    std::cout << "Max absolute error: " << max_error << std::endl;

    // Check if top predictions match
    bool topMatch = (naiveTopClass == quantTopClass);
    std::cout << "Top prediction match: " << (topMatch ? "YES" : "NO") << std::endl;

    // Calculate total INT8 time
    double totalQuantTime = 0.0;
    for (size_t i = 0; i < layerTimes.size(); i++) {
        totalQuantTime += layerTimes[i];
    }

    // Performance comparison
    logInfo("\n--- Step 4: Layer-wise timing breakdown ---");
    std::cout << "\nINT8 QUANTIZED Layer Timing:" << std::endl;
    std::cout << "Layer |      Name      |   Time (ms)  | % of Total" << std::endl;
    std::cout << "------|----------------|--------------|----------" << std::endl;
    
    for (size_t i = 0; i < model.getNumLayers(); i++) {
        double percentage = (layerTimes[i] / totalQuantTime) * 100.0;
        std::cout << std::setw(5) << i << " | "
                  << std::setw(14) << std::left << layerNames[i] << " | "
                  << std::setw(12) << std::fixed << std::setprecision(6) << layerTimes[i] << " | "
                  << std::setw(8) << std::fixed << std::setprecision(2) << percentage << "%" << std::endl;
    }
    
    std::cout << "------|----------------|--------------|----------" << std::endl;
    std::cout << "TOTAL |                | "
              << std::setw(12) << std::fixed << std::setprecision(6) << totalQuantTime << " | "
              << "100.00%" << std::endl;

    // Performance comparison summary
    logInfo("\n--- Step 5: Overall performance comparison ---");
    fp32 speedup = naiveTimer.milliseconds / totalQuantTime;
    std::cout << "FP32 NAIVE time:     " << naiveTimer.milliseconds << " ms" << std::endl;
    std::cout << "INT8 QUANTIZED time: " << totalQuantTime << " ms" << std::endl;
    std::cout << "Speedup:             " << speedup << "x" << std::endl;

    if (speedup > 1.0f) {
        std::cout << "INT8 is FASTER by " << ((speedup - 1.0f) * 100.0f) << "%" << std::endl;
    } else {
        std::cout << "INT8 is SLOWER by " << ((1.0f - speedup) * 100.0f) << "%" << std::endl;
    }

    // Print detailed probability comparison
    std::cout << "\nDetailed probability comparison:" << std::endl;
    std::cout << "Class | FP32        | INT8        | Diff" << std::endl;
    std::cout << "------|-------------|-------------|-------------" << std::endl;
    for (size_t i = 0; i < 10; i++) {
        fp32 naiveProb = naiveOutputCopy.get<fp32>(i) * 100.0f;
        fp32 quantProb = quantOutput.get<fp32>(i) * 100.0f;
        fp32 diff = naiveProb - quantProb;
        std::cout << std::setw(5) << i << " | "
                  << std::setw(10) << std::fixed << std::setprecision(4) << naiveProb << "% | "
                  << std::setw(10) << std::fixed << std::setprecision(4) << quantProb << "% | "
                  << std::setw(10) << std::fixed << std::setprecision(4) << diff << "%" << std::endl;
    }
}

void runQuantizedInferenceTest(const Model& model, const LayerData& inputData) {
    logInfo("========================================");
    logInfo("  INT8 QUANTIZED INFERENCE TEST");
    logInfo("========================================");

    // First run NAIVE for baseline
    logInfo("\n--- Step 1: Running FP32 NAIVE baseline ---");
    Timer naiveTimer("FP32 NAIVE Inference");
    naiveTimer.start();
    const LayerData& naiveOutput = model.inference(inputData, Layer::InfType::NAIVE);
    naiveTimer.stop();

    // Copy NAIVE output for comparison
    LayerData naiveOutputCopy(naiveOutput);

    // Print NAIVE top prediction
    const char* instrumentNames[] = {
        "Cello", "Clarinet", "Flute", "Acoustic Guitar", "Electric Guitar",
        "Organ", "Piano", "Saxophone", "Trumpet", "Violin"
    };

    size_t naiveTopClass = 0;
    fp32 naiveTopProb = naiveOutput.get<fp32>(0);
    for (size_t i = 1; i < 10; i++) {
        if (naiveOutput.get<fp32>(i) > naiveTopProb) {
            naiveTopProb = naiveOutput.get<fp32>(i);
            naiveTopClass = i;
        }
    }
    std::cout << "FP32 NAIVE Top prediction: " << instrumentNames[naiveTopClass]
              << " (" << (naiveTopProb * 100.0f) << "%)" << std::endl;

    // Now run QUANTIZED
    logInfo("\n--- Step 2: Running INT8 QUANTIZED inference ---");
    Timer quantTimer("INT8 QUANTIZED Inference");
    quantTimer.start();
    const LayerData& quantOutput = model.inference(inputData, Layer::InfType::QUANTIZED);
    quantTimer.stop();

    // Print QUANTIZED top prediction
    size_t quantTopClass = 0;
    fp32 quantTopProb = quantOutput.get<fp32>(0);
    for (size_t i = 1; i < 10; i++) {
        if (quantOutput.get<fp32>(i) > quantTopProb) {
            quantTopProb = quantOutput.get<fp32>(i);
            quantTopClass = i;
        }
    }
    std::cout << "INT8 QUANTIZED Top prediction: " << instrumentNames[quantTopClass]
              << " (" << (quantTopProb * 100.0f) << "%)" << std::endl;

    // Compare outputs
    logInfo("\n--- Step 3: Comparing FP32 vs INT8 results ---");

    // Calculate cosine similarity
    fp32 cosine_sim = naiveOutputCopy.compare<fp32>(quantOutput);
    std::cout << "Cosine similarity: " << (cosine_sim * 100.0f) << "%" << std::endl;

    // Calculate max absolute error
    fp32 max_error = 0.0f;
    for (size_t i = 0; i < 10; i++) {
        fp32 error = std::abs(naiveOutputCopy.get<fp32>(i) - quantOutput.get<fp32>(i));
        max_error = std::max(max_error, error);
    }
    std::cout << "Max absolute error: " << max_error << std::endl;

    // Check if top predictions match
    bool topMatch = (naiveTopClass == quantTopClass);
    std::cout << "Top prediction match: " << (topMatch ? "YES" : "NO") << std::endl;

    // Performance comparison
    logInfo("\n--- Step 4: Performance comparison ---");
    fp32 speedup = naiveTimer.milliseconds / quantTimer.milliseconds;
    std::cout << "FP32 NAIVE time:     " << naiveTimer.milliseconds << " ms" << std::endl;
    std::cout << "INT8 QUANTIZED time: " << quantTimer.milliseconds << " ms" << std::endl;
    std::cout << "Speedup:             " << speedup << "x" << std::endl;

    if (speedup > 1.0f) {
        std::cout << "INT8 is FASTER by " << ((speedup - 1.0f) * 100.0f) << "%" << std::endl;
    } else {
        std::cout << "INT8 is SLOWER by " << ((1.0f - speedup) * 100.0f) << "%" << std::endl;
    }

    // Print detailed probability comparison
    std::cout << "\nDetailed probability comparison:" << std::endl;
    std::cout << "Class | FP32        | INT8        | Diff" << std::endl;
    std::cout << "------|-------------|-------------|-------------" << std::endl;
    for (size_t i = 0; i < 10; i++) {
        fp32 naiveProb = naiveOutputCopy.get<fp32>(i) * 100.0f;
        fp32 quantProb = quantOutput.get<fp32>(i) * 100.0f;
        fp32 diff = naiveProb - quantProb;
        std::cout << std::setw(5) << i << " | "
                  << std::setw(10) << std::fixed << std::setprecision(4) << naiveProb << "% | "
                  << std::setw(10) << std::fixed << std::setprecision(4) << quantProb << "% | "
                  << std::setw(10) << std::fixed << std::setprecision(4) << diff << "%" << std::endl;
    }
}
void runLayerByLayerDiagnostic(const Model& model, const LayerData& inputData) {
    logInfo("========================================");
    logInfo("  LAYER-BY-LAYER QUANTIZATION DIAGNOSTIC");
    logInfo("========================================");
    
    const char* layerNames[] = {
        "Layer 0: conv1_1",
        "Layer 1: bn1_1",
        "Layer 2: conv1_2",
        "Layer 3: bn1_2",
        "Layer 4: pool1",
        "Layer 5: conv2_1",
        "Layer 6: bn2_1",
        "Layer 7: conv2_2",
        "Layer 8: bn2_2",
        "Layer 9: pool2",
        "Layer 10: conv3_1",
        "Layer 11: bn3_1",
        "Layer 12: conv3_2",
        "Layer 13: bn3_2",
        "Layer 14: pool3",
        "Layer 15: flatten",
        "Layer 16: fc1",
        "Layer 17: bn_fc1",
        "Layer 18: fc2",
        "Layer 19: softmax"
    };

    // Run FP32 inference and capture all intermediate outputs
    std::cout << "\n[DIAGNOSTIC] Running FP32 NAIVE inference...\n" << std::endl;
    model.inferenceLayer(inputData, 0, Layer::InfType::NAIVE);
    
    std::vector<LayerData> fp32_outputs;
    fp32_outputs.push_back(model[0].getOutputData());
    
    for (size_t i = 1; i < model.getNumLayers(); i++) {
        model.inferenceLayer(fp32_outputs.back(), i, Layer::InfType::NAIVE);
        fp32_outputs.push_back(model[i].getOutputData());
    }
    
    // Run INT8 QUANTIZED inference and capture all intermediate outputs
    std::cout << "\n[DIAGNOSTIC] Running INT8 QUANTIZED inference...\n" << std::endl;
    model.inferenceLayer(inputData, 0, Layer::InfType::QUANTIZED);
    
    std::vector<LayerData> int8_outputs;
    int8_outputs.push_back(model[0].getOutputData());
    
    for (size_t i = 1; i < model.getNumLayers(); i++) {
        model.inferenceLayer(int8_outputs.back(), i, Layer::InfType::QUANTIZED);
        int8_outputs.push_back(model[i].getOutputData());
    }
    
    // Compare layer outputs
    std::cout << "\n[DIAGNOSTIC] Comparing layer outputs:\n" << std::endl;
    std::cout << "Layer | Cosine Sim | Max Error | Min(FP32) | Max(FP32) | Min(INT8) | Max(INT8)" << std::endl;
    std::cout << "------|-----------|-----------|-----------|-----------|-----------|----------" << std::endl;
    
    int divergence_layer = -1;
    for (size_t i = 0; i < model.getNumLayers(); i++) {
        fp32 cosine_sim = fp32_outputs[i].compare<fp32>(int8_outputs[i]);
        
        fp32 fp32_min = fp32_outputs[i].get<fp32>(0);
        fp32 fp32_max = fp32_outputs[i].get<fp32>(0);
        fp32 int8_min = int8_outputs[i].get<fp32>(0);
        fp32 int8_max = int8_outputs[i].get<fp32>(0);
        fp32 max_error = 0.0f;
        
        for (size_t j = 0; j < fp32_outputs[i].getParams().flat_count(); j++) {
            fp32 fp32_val = fp32_outputs[i].get<fp32>(j);
            fp32 int8_val = int8_outputs[i].get<fp32>(j);
            
            fp32_min = std::min(fp32_min, fp32_val);
            fp32_max = std::max(fp32_max, fp32_val);
            int8_min = std::min(int8_min, int8_val);
            int8_max = std::max(int8_max, int8_val);
            
            max_error = std::max(max_error, std::abs(fp32_val - int8_val));
        }
        
        std::cout << std::setw(5) << i << " | "
                  << std::setw(9) << std::fixed << std::setprecision(2) << (cosine_sim * 100.0f) << "% | "
                  << std::setw(9) << std::fixed << std::setprecision(4) << max_error << " | "
                  << std::setw(9) << std::fixed << std::setprecision(4) << fp32_min << " | "
                  << std::setw(9) << std::fixed << std::setprecision(4) << fp32_max << " | "
                  << std::setw(9) << std::fixed << std::setprecision(4) << int8_min << " | "
                  << std::setw(9) << std::fixed << std::setprecision(4) << int8_max << std::endl;
        
        // Flag first layer with low cosine similarity
        if (cosine_sim < 0.9f && divergence_layer == -1) {
            divergence_layer = i;
            std::cout << "  >>> FIRST DIVERGENCE at " << layerNames[i] << " (cosine sim: " << (cosine_sim*100.0f) << "%)" << std::endl;
        }
    }
    
    if (divergence_layer >= 0) {
        std::cout << "\n[DIAGNOSTIC] Quantization starts diverging at layer " << divergence_layer 
                  << " (" << layerNames[divergence_layer] << ")" << std::endl;
        std::cout << "[DIAGNOSTIC] This is likely where the quantization formula is incorrect." << std::endl;
    } else {
        std::cout << "\n[DIAGNOSTIC] All layers match well! Quantization is working correctly." << std::endl;
    }
}

void runAllLayerTests(const Model& model, const Path& basePath, const LayerData& inputData) {
    logInfo("--- Running All Layer Tests ---");

    // Test all layers (0-12 for AudioCNN_IRMAS)
    size_t numLayers = model.getNumLayers();
    for (std::size_t layerNum = 0; layerNum < numLayers; ++layerNum) {
        runLayerTest(layerNum, model, basePath, inputData);
    }
}

void runBatchInferenceTest(const Model& model, const Path& dataPath, int numSamples = 10) {
    logInfo("========================================");
    logInfo("  BATCH INFERENCE TEST");
    logInfo("========================================");

    const char* instrumentNames[] = {
        "Cello", "Clarinet", "Flute", "Acoustic Guitar", "Electric Guitar",
        "Organ", "Piano", "Saxophone", "Trumpet", "Violin"
    };

    const char* instrumentCodes[] = {
        "cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio"
    };

    // Load metadata from JSON file (simplified - manual parsing for now)
    // In production, we'd use a JSON parser library
    std::cout << "\nLoading batch test metadata...\n" << std::endl;

    BatchTestResults results;
    results.total_samples = numSamples;

    // Process each test sample
    for (int i = 0; i < numSamples; i++) {
        std::stringstream filename;
        filename << "test_input_" << std::setfill('0') << std::setw(3) << i << ".bin";

        Path inputPath = dataPath / filename.str().c_str();

        // Check if file exists
        std::ifstream testFile(inputPath, std::ios::binary);
        if (!testFile.is_open()) {
            std::cout << "   Sample " << i << ": File not found - " << filename.str() << std::endl;
            continue;
        }
        testFile.close();

        // Load test input
        LayerData testInput({sizeof(fp32), {128, 128, 1}, inputPath});
        try {
            testInput.loadData();
        } catch (const std::exception& e) {
            std::cout << "   Sample " << i << ": Failed to load - " << e.what() << std::endl;
            continue;
        }

        // Determine true label from filename pattern (1 sample per class, sequential)
        int true_label = i;  // One sample per class: 0->cel, 1->cla, ..., 9->vio
        if (true_label >= 10) true_label = 9;  // Clamp to valid range

        results.true_labels.push_back(true_label);

        // Run FP32 NAIVE inference
        Timer fp32Timer("FP32");
        fp32Timer.start();
        const LayerData& fp32Output = model.inference(testInput, Layer::InfType::NAIVE);
        fp32Timer.stop();
        results.fp32_times.push_back(fp32Timer.milliseconds);

        // Get FP32 top prediction
        int fp32TopClass = 0;
        fp32 fp32TopProb = fp32Output.get<fp32>(0);
        for (int c = 1; c < 10; c++) {
            if (fp32Output.get<fp32>(c) > fp32TopProb) {
                fp32TopProb = fp32Output.get<fp32>(c);
                fp32TopClass = c;
            }
        }
        results.predicted_labels_fp32.push_back(fp32TopClass);

        // Track FP32 top-1 accuracy
        if (fp32TopClass == true_label) {
            results.correct_top1_fp32++;
        }

        // Track FP32 top-5 accuracy
        std::vector<std::pair<fp32, int>> fp32_predictions;
        for (int c = 0; c < 10; c++) {
            fp32_predictions.push_back({fp32Output.get<fp32>(c), c});
        }
        std::sort(fp32_predictions.begin(), fp32_predictions.end(), std::greater<std::pair<fp32, int>>());
        for (int k = 0; k < 5; k++) {
            if (fp32_predictions[k].second == true_label) {
                results.correct_top5_fp32++;
                break;
            }
        }

        // Copy FP32 output for comparison
        LayerData fp32OutputCopy(fp32Output);

        // Run INT8 QUANTIZED inference
        Timer int8Timer("INT8");
        int8Timer.start();
        const LayerData& int8Output = model.inference(testInput, Layer::InfType::QUANTIZED);
        int8Timer.stop();
        results.int8_times.push_back(int8Timer.milliseconds);

        // Get INT8 top prediction
        int int8TopClass = 0;
        fp32 int8TopProb = int8Output.get<fp32>(0);
        for (int c = 1; c < 10; c++) {
            if (int8Output.get<fp32>(c) > int8TopProb) {
                int8TopProb = int8Output.get<fp32>(c);
                int8TopClass = c;
            }
        }
        results.predicted_labels_int8.push_back(int8TopClass);

        // Calculate cosine similarity
        fp32 cosine_sim = fp32OutputCopy.compare<fp32>(int8Output);
        results.cosine_similarities.push_back(cosine_sim);

        // Update INT8 accuracy counters
        if (int8TopClass == true_label) {
            results.correct_top1_int8++;
        }

        // Check INT8 top-5 accuracy
        std::vector<std::pair<fp32, int>> predictions;
        for (int c = 0; c < 10; c++) {
            predictions.push_back({int8Output.get<fp32>(c), c});
        }
        std::sort(predictions.begin(), predictions.end(), std::greater<std::pair<fp32, int>>());

        bool in_top5 = false;
        for (int k = 0; k < 5; k++) {
            if (predictions[k].second == true_label) {
                in_top5 = true;
                results.correct_top5_int8++;
                break;
            }
        }

        // Print sample result
        const char* correctMark = (int8TopClass == true_label) ? "[OK]" : "[XX]";
        std::cout << "  " << correctMark << " Sample " << std::setw(3) << i
                  << " | True: " << std::setw(3) << instrumentCodes[true_label]
                  << " | Pred: " << std::setw(3) << instrumentCodes[int8TopClass]
                  << " | Conf: " << std::fixed << std::setprecision(2) << std::setw(6) << (int8TopProb * 100.0f) << "%"
                  << " | CosSim: " << std::setw(6) << (cosine_sim * 100.0f) << "%"
                  << " | FP32: " << std::setw(5) << std::fixed << std::setprecision(0) << fp32Timer.milliseconds << "ms"
                  << " | INT8: " << std::setw(5) << int8Timer.milliseconds << "ms" << std::endl;
    }

    // Calculate statistics
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "BATCH TEST SUMMARY" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Accuracy metrics
    int valid_samples = results.true_labels.size();

    // FP32 Accuracy
    double fp32_top1_accuracy = (valid_samples > 0) ? (100.0 * results.correct_top1_fp32 / valid_samples) : 0.0;
    double fp32_top5_accuracy = (valid_samples > 0) ? (100.0 * results.correct_top5_fp32 / valid_samples) : 0.0;

    // INT8 Accuracy
    double int8_top1_accuracy = (valid_samples > 0) ? (100.0 * results.correct_top1_int8 / valid_samples) : 0.0;
    double int8_top5_accuracy = (valid_samples > 0) ? (100.0 * results.correct_top5_int8 / valid_samples) : 0.0;

    std::cout << "\nACCURACY METRICS:" << std::endl;
    std::cout << "  Total samples: " << valid_samples << " / " << numSamples << std::endl;

    std::cout << "\nFP32 ACCURACY:" << std::endl;
    std::cout << "  Top-1 Accuracy: " << results.correct_top1_fp32 << " / " << valid_samples
              << " (" << std::fixed << std::setprecision(2) << fp32_top1_accuracy << "%)" << std::endl;
    std::cout << "  Top-5 Accuracy: " << results.correct_top5_fp32 << " / " << valid_samples
              << " (" << std::fixed << std::setprecision(2) << fp32_top5_accuracy << "%)" << std::endl;

    std::cout << "\nINT8 ACCURACY:" << std::endl;
    std::cout << "  Top-1 Accuracy: " << results.correct_top1_int8 << " / " << valid_samples
              << " (" << std::fixed << std::setprecision(2) << int8_top1_accuracy << "%)" << std::endl;
    std::cout << "  Top-5 Accuracy: " << results.correct_top5_int8 << " / " << valid_samples
              << " (" << std::fixed << std::setprecision(2) << int8_top5_accuracy << "%)" << std::endl;

    std::cout << "\nQUANTIZATION ACCURACY LOSS:" << std::endl;
    std::cout << "  Top-1 Loss: " << std::fixed << std::setprecision(2)
              << (fp32_top1_accuracy - int8_top1_accuracy) << "%" << std::endl;
    std::cout << "  Top-5 Loss: " << std::fixed << std::setprecision(2)
              << (fp32_top5_accuracy - int8_top5_accuracy) << "%" << std::endl;

    // Timing statistics - FP32
    if (!results.fp32_times.empty()) {
        double fp32_mean = std::accumulate(results.fp32_times.begin(), results.fp32_times.end(), 0.0) / results.fp32_times.size();
        double fp32_min = *std::min_element(results.fp32_times.begin(), results.fp32_times.end());
        double fp32_max = *std::max_element(results.fp32_times.begin(), results.fp32_times.end());

        double fp32_variance = 0.0;
        for (double t : results.fp32_times) {
            fp32_variance += (t - fp32_mean) * (t - fp32_mean);
        }
        fp32_variance /= results.fp32_times.size();
        double fp32_std = std::sqrt(fp32_variance);

        std::cout << "\nFP32 RUNTIME STATISTICS:" << std::endl;
        std::cout << "  Mean:   " << std::fixed << std::setprecision(2) << fp32_mean << " ms" << std::endl;
        std::cout << "  Std:    " << fp32_std << " ms" << std::endl;
        std::cout << "  Min:    " << fp32_min << " ms" << std::endl;
        std::cout << "  Max:    " << fp32_max << " ms" << std::endl;
    }

    // Timing statistics - INT8
    if (!results.int8_times.empty()) {
        double int8_mean = std::accumulate(results.int8_times.begin(), results.int8_times.end(), 0.0) / results.int8_times.size();
        double int8_min = *std::min_element(results.int8_times.begin(), results.int8_times.end());
        double int8_max = *std::max_element(results.int8_times.begin(), results.int8_times.end());

        double int8_variance = 0.0;
        for (double t : results.int8_times) {
            int8_variance += (t - int8_mean) * (t - int8_mean);
        }
        int8_variance /= results.int8_times.size();
        double int8_std = std::sqrt(int8_variance);

        std::cout << "\nINT8 RUNTIME STATISTICS:" << std::endl;
        std::cout << "  Mean:   " << std::fixed << std::setprecision(2) << int8_mean << " ms" << std::endl;
        std::cout << "  Std:    " << int8_std << " ms" << std::endl;
        std::cout << "  Min:    " << int8_min << " ms" << std::endl;
        std::cout << "  Max:    " << int8_max << " ms" << std::endl;

        // Speedup
        if (!results.fp32_times.empty()) {
            double fp32_mean = std::accumulate(results.fp32_times.begin(), results.fp32_times.end(), 0.0) / results.fp32_times.size();
            double speedup = fp32_mean / int8_mean;
            std::cout << "\nSPEEDUP (FP32 → INT8):" << std::endl;
            std::cout << "  Average Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
            std::cout << "  Performance Gain: " << std::fixed << std::setprecision(2) << ((speedup - 1.0) * 100.0) << "%" << std::endl;
        }
    }

    // Cosine similarity statistics
    if (!results.cosine_similarities.empty()) {
        double cos_mean = std::accumulate(results.cosine_similarities.begin(), results.cosine_similarities.end(), 0.0) / results.cosine_similarities.size();
        double cos_min = *std::min_element(results.cosine_similarities.begin(), results.cosine_similarities.end());
        double cos_max = *std::max_element(results.cosine_similarities.begin(), results.cosine_similarities.end());

        double cos_variance = 0.0;
        for (double s : results.cosine_similarities) {
            cos_variance += (s - cos_mean) * (s - cos_mean);
        }
        cos_variance /= results.cosine_similarities.size();
        double cos_std = std::sqrt(cos_variance);

        std::cout << "\nQUANTIZATION QUALITY (Cosine Similarity):" << std::endl;
        std::cout << "  Mean:   " << std::fixed << std::setprecision(4) << (cos_mean * 100.0) << "%" << std::endl;
        std::cout << "  Std:    " << std::fixed << std::setprecision(4) << (cos_std * 100.0) << "%" << std::endl;
        std::cout << "  Min:    " << std::fixed << std::setprecision(4) << (cos_min * 100.0) << "%" << std::endl;
        std::cout << "  Max:    " << std::fixed << std::setprecision(4) << (cos_max * 100.0) << "%" << std::endl;
    }

    // Confusion matrix (simplified - just show per-class accuracy)
    std::cout << "\nPER-CLASS ACCURACY:" << std::endl;
    std::cout << "Class | Name             | Correct | Total | Accuracy" << std::endl;
    std::cout << "------|------------------|---------|-------|----------" << std::endl;

    int class_counts[10] = {0};
    int class_correct[10] = {0};

    for (size_t i = 0; i < results.true_labels.size(); i++) {
        int true_label = results.true_labels[i];
        int pred_label = results.predicted_labels_int8[i];

        class_counts[true_label]++;
        if (true_label == pred_label) {
            class_correct[true_label]++;
        }
    }

    for (int c = 0; c < 10; c++) {
        if (class_counts[c] > 0) {
            double class_acc = (100.0 * class_correct[c]) / class_counts[c];
            std::cout << std::setw(5) << c << " | "
                      << std::setw(16) << std::left << instrumentNames[c] << std::right << " | "
                      << std::setw(7) << class_correct[c] << " | "
                      << std::setw(5) << class_counts[c] << " | "
                      << std::setw(7) << std::fixed << std::setprecision(2) << class_acc << "%" << std::endl;
        }
    }

    std::cout << "\n" << std::string(70, '=') << std::endl;
}

void runTests() {
    logInfo("========================================");
    logInfo("  AudioCNN_IRMAS Model Testing");
    logInfo("  Musical Instrument Classification");
    logInfo("========================================");
    
    // Base paths for audio model
    Path basePath("data");
    Path modelPath = basePath / "model_weights_improved";
    Path featureMapsPath = basePath / "feature_maps_improved";
    
    // Build the AudioCNN_IRMAS model
    Model model = buildAudioCNN_IRMAS(modelPath);
    model.allocLayers();
    
    // Load a test mel-spectrogram (128x128x1)
    logInfo("Loading test mel-spectrogram...");
    LayerData melSpec({sizeof(fp32), {128, 128, 1}, basePath / "test_input.bin"});
    melSpec.loadData();
    logInfo("Test input loaded successfully!");
    
    // Run layer-by-layer tests
    //runAllLayerTests(model, featureMapsPath, melSpec);

    // Run full inference test
    //runInferenceTest(model, melSpec);

    // Run quantized inference test with layer-by-layer diagnostic
    std::cout << "\n\n";
    runLayerByLayerDiagnostic(model, melSpec);

    std::cout << "\n\n";
    runQuantizedInferenceTestWithLayerTiming(model, melSpec);

    // Run batch inference test on multiple validation samples
    std::cout << "\n\n";
    runBatchInferenceTest(model, basePath, 10);

    // Clean up
    model.freeLayers();

    std::cout << "\n\n----- ML::runTests() COMPLETE -----\n";
}

} // namespace ML

#ifdef ZEDBOARD
extern "C"
int main() {
    try {
        static FATFS fatfs;
        if (f_mount(&fatfs, "/", 1) != FR_OK) {
            throw std::runtime_error("Failed to mount SD card. Is it plugged in?");
        }
        ML::runTests();
    } catch (const std::exception& e) {
        std::cerr << "\n\n----- EXCEPTION THROWN -----\n" << e.what() << '\n';
    }
    std::cout << "\n\n----- STARTING FILE TRANSFER SERVER -----\n";
    FileServer::start_file_transfer_server();
}
#else
int main() {
    ML::runTests();
}
#endif

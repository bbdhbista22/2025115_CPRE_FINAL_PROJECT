#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

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

        // Load the expected output for this specific layer
        std::string expectedFileName = "layer_" + std::to_string(layerNum) + "_output.bin";
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

void runAllLayerTests(const Model& model, const Path& basePath, const LayerData& inputData) {
    logInfo("--- Running All Layer Tests ---");
    
    // Test all layers (0-12 for AudioCNN_IRMAS)
    size_t numLayers = model.getNumLayers();
    for (std::size_t layerNum = 0; layerNum < numLayers; ++layerNum) {
        runLayerTest(layerNum, model, basePath, inputData);
    }
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
    runAllLayerTests(model, featureMapsPath, melSpec);
    
    // Run full inference test
    runInferenceTest(model, melSpec);
    
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

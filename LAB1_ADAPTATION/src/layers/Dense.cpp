#include "Dense.h"

#include <iostream>
#include <algorithm>
#include <thread>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML
{
    // ==========================================================================
    // INT8 QUANTIZATION - CALIBRATION STATISTICS FOR DENSE LAYERS
    // ==========================================================================

    // Reuse the calibration data from Convolutional layer
    extern std::map<std::string, struct CalibrationStats> calibration_data;
    extern bool calibration_loaded;

    // Define CalibrationStats if not already defined
    #ifndef CALIBRATION_STATS_DEFINED
    #define CALIBRATION_STATS_DEFINED
    struct CalibrationStats {
        fp32 min, max, mean, std;
        fp32 Si;  // Scale factor
        i8 zi;    // Zero point
    };
    #endif

    // Helper function to extract layer name from file path
    std::string extractDenseLayerNameFromPath(const std::string& filepath) {
        size_t last_slash = filepath.find_last_of("/\\");
        std::string filename = (last_slash != std::string::npos) ? filepath.substr(last_slash + 1) : filepath;
        
        // Remove "_weights.bin" suffix
        size_t underscore_pos = filename.find("_weights.bin");
        if (underscore_pos != std::string::npos) {
            return filename.substr(0, underscore_pos);
        }
        return filename;
    }

    void DenseLayer::computeNaive(const LayerData &dataIn) const
    {
        //const auto &inputDims = getInputParams().dims;   // Can be [H, W, C] or [features] 
        //const auto &outputDims = getOutputParams().dims; // Expected: [output_features]
        const auto &weightDims = getWeightParams().dims; // Expected: [input_features, output_features]

        // Calculate total input features by flattening all input dimensions
        size_t totalInputFeatures = getInputParams().flat_count();
        size_t outputSize = getOutputParams().flat_count();

        // Validate dimensions
        size_t expectedInputFeatures = weightDims[0];  // First dimension of weight matrix
        size_t expectedOutputFeatures = weightDims[1]; // Second dimension of weight matrix
        
        if (totalInputFeatures != expectedInputFeatures) {
            std::cerr << "Dense layer input size mismatch: got " << totalInputFeatures 
                      << ", expected " << expectedInputFeatures << std::endl;
            return;
        }
        
        if (outputSize != expectedOutputFeatures) {
            std::cerr << "Dense layer output size mismatch: got " << outputSize 
                      << ", expected " << expectedOutputFeatures << std::endl;
            return;
        }

        const LayerData& weights = getWeightData();
        LayerData& output = getOutputData();
        const LayerData& bias = getBiasData();

        // Dense layer computation: output = input * weights + bias
        // Input is treated as flattened regardless of original dimensions
        #pragma omp parallel for
        for (size_t out_idx = 0; out_idx < outputSize; out_idx++)
        {
            fp32 sum = bias.get<fp32>(out_idx);

            for (size_t in_idx = 0; in_idx < totalInputFeatures; in_idx++)
            {
                // Weight matrix: [input_features, output_features]
                size_t weightIdx = in_idx * outputSize + out_idx;

                sum += dataIn.get<fp32>(in_idx) * weights.get<fp32>(weightIdx);
            }

            // Store result in output
            output.get<fp32>(out_idx) = sum;
        }
    }

    void DenseLayer::computeThreaded(const LayerData& dataIn) const {
        // For simplicity, use naive implementation with thread hints
        // TODO: Implement actual threading
        computeNaive(dataIn);
    }

    void DenseLayer::computeTiled(const LayerData& dataIn) const {
        // For simplicity, use naive implementation 
        // TODO: Implement tiled matrix multiplication
        computeNaive(dataIn);
    }

    void DenseLayer::computeSIMD(const LayerData& dataIn) const {
        // For simplicity, use naive implementation
        // TODO: Implement SIMD optimized matrix multiplication
        computeNaive(dataIn);
    }

    void DenseLayer::computeQuantized(const LayerData& dataIn) const {
        // INT8 Quantized Dense Layer Implementation using calibration stats

        if (!calibration_loaded) {
            logError("Calibration stats not loaded, falling back to NAIVE");
            computeNaive(dataIn);
            return;
        }

        const auto &weightDims = getWeightParams().dims; // Expected: [input_features, output_features]

        // Calculate total input features by flattening all input dimensions
        size_t totalInputFeatures = getInputParams().flat_count();
        size_t outputSize = getOutputParams().flat_count();

        // Validate dimensions
        size_t expectedInputFeatures = weightDims[0];  // First dimension of weight matrix
        size_t expectedOutputFeatures = weightDims[1]; // Second dimension of weight matrix

        if (totalInputFeatures != expectedInputFeatures) {
            std::cerr << "Dense layer input size mismatch: got " << totalInputFeatures
                      << ", expected " << expectedInputFeatures << std::endl;
            return;
        }

        if (outputSize != expectedOutputFeatures) {
            std::cerr << "Dense layer output size mismatch: got " << outputSize
                      << ", expected " << expectedOutputFeatures << std::endl;
            return;
        }

        const LayerData& weights = getWeightData();
        LayerData& output = getOutputData();
        const LayerData& bias = getBiasData();

        // --- Use Calibration Stats ---
        // Extract layer name from weight file path to determine calibration stats
        // E.g., "fc1_weights.bin" -> "fc1"
        std::string layer_name = extractDenseLayerNameFromPath(getWeightParams().filePath);

        // Map layer name to input calibration stats
        // Architecture: flatten -> fc1 -> bn_fc1 -> fc2
        std::string input_stats_name;
        if (layer_name == "fc1") {
            input_stats_name = "flatten";  // FC1 gets input from flatten
        } else if (layer_name == "fc2") {
            input_stats_name = "bn_fc1";   // FC2 gets input from bn_fc1
        } else {
            input_stats_name = "bn_fc1";   // Fallback
        }

        auto input_stats_it = calibration_data.find(input_stats_name);
        if (input_stats_it == calibration_data.end()) {
            logError("No calibration stats found for: " + input_stats_name);
            logError("Falling back to dynamic quantization");

            // Fallback to dynamic quantization
            fp32 input_max = 0.0f;
            for (size_t i = 0; i < totalInputFeatures; i++) {
                fp32 val = std::abs(dataIn.get<fp32>(i));
                if (val > input_max) input_max = val;
            }
            //fp32 Si = (input_max > 0) ? (127.0f / input_max) : 1.0f;
            

            // std::cout << "[QUANT] " << layer_name << " - Dynamic fallback" << std::endl;
            // std::cout << "[QUANT]   Input max: " << input_max << ", scale (Si): " << Si << std::endl;

            fp32 weight_max = 0.0f;
            for (size_t i = 0; i < totalInputFeatures * outputSize; i++) {
                fp32 val = std::abs(weights.get<fp32>(i));
                if (val > weight_max) weight_max = val;
            }
            //fp32 Sw = (weight_max > 0) ? (127.0f / weight_max) : 1.0f;
            //fp32 Sb = Si * Sw;

            // std::cout << "[QUANT]   Weight max: " << weight_max << ", scale (Sw): " << Sw << std::endl;
            // std::cout << "[QUANT]   Bias scale (Sb): " << Sb << std::endl;

            //dense_layer_count++;
            computeNaive(dataIn);
            return;
        }

        const CalibrationStats &input_stats = input_stats_it->second;
        fp32 Si = input_stats.Si;
        i8 zi = input_stats.zi;

        // std::cout << "[QUANT] " << layer_name << " - Using calibration: " << input_stats_name << std::endl;
        // std::cout << "[QUANT]   Input scale (Si): " << Si << ", zero-point (zi): " << static_cast<int>(zi) << std::endl;
        // std::cout << "[QUANT]   Calibration range: [" << input_stats.min << ", " << input_stats.max << "], mean: " << input_stats.mean << std::endl;

        //dense_layer_count++;

        // Calculate weight scale
        fp32 weight_max = 0.0f;
        for (size_t i = 0; i < totalInputFeatures * outputSize; i++) {
            fp32 val = std::abs(weights.get<fp32>(i));
            if (val > weight_max) weight_max = val;
        }
        fp32 Sw = (weight_max > 0) ? (127.0f / weight_max) : 1.0f;

        // Bias scale = Si * Sw
        fp32 Sb = Si * Sw;

        // std::cout << "[QUANT]   Weight max: " << weight_max << ", scale (Sw): " << Sw << std::endl;
        // std::cout << "[QUANT]   Bias scale (Sb): " << Sb << std::endl;

        // Log sample input values before quantization
        // std::cout << "[QUANT]   Sample FP32 inputs: ";
        for (size_t i = 0; i < std::min(size_t(5), totalInputFeatures); i++) {
            std::cout << dataIn.get<fp32>(i) << " ";
        }
        std::cout << std::endl;

        // --- Quantize inputs ---
        std::vector<i8> input_quantized(totalInputFeatures);
        for (size_t i = 0; i < totalInputFeatures; i++) {
            fp32 val = dataIn.get<fp32>(i);
            i32 quantized = static_cast<i32>(std::round(val * Si)) + zi;
            input_quantized[i] = static_cast<i8>(std::max(-128, std::min(127, quantized)));
        }

        // --- Quantize weights ---
        std::vector<i8> weights_quantized(totalInputFeatures * outputSize);
        for (size_t i = 0; i < totalInputFeatures * outputSize; i++) {
            fp32 val = weights.get<fp32>(i);
            i32 quantized = static_cast<i32>(std::round(val * Sw));
            weights_quantized[i] = static_cast<i8>(std::max(-128, std::min(127, quantized)));
        }

        // --- Quantize biases ---
        std::vector<i32> bias_quantized(outputSize);
        for (size_t i = 0; i < outputSize; i++) {
            fp32 val = bias.get<fp32>(i);
            bias_quantized[i] = static_cast<i32>(std::round(val * Sb));
        }

        // Log sample quantized values
        // std::cout << "[QUANT]   Sample INT8 inputs: ";
        for (size_t i = 0; i < std::min(size_t(5), totalInputFeatures); i++) {
            std::cout << static_cast<int>(input_quantized[i]) << " ";
        }
        std::cout << std::endl;

        // std::cout << "[QUANT]   Sample INT8 weights: ";
        for (size_t i = 0; i < std::min(size_t(5), totalInputFeatures * outputSize); i++) {
            std::cout << static_cast<int>(weights_quantized[i]) << " ";
        }
        std::cout << std::endl;

        // Precompute sum of quantized weights for each output feature (for zero-point correction)
        std::vector<i32> weight_sum_per_output(outputSize, 0);
        for (size_t out_idx = 0; out_idx < outputSize; out_idx++) {
            for (size_t in_idx = 0; in_idx < totalInputFeatures; in_idx++) {
                size_t weightIdx = in_idx * outputSize + out_idx;
                weight_sum_per_output[out_idx] += static_cast<i32>(weights_quantized[weightIdx]);
            }
        }

        // --- INT8 Matrix-Vector Multiply ---
        #pragma omp parallel for
        for (size_t out_idx = 0; out_idx < outputSize; out_idx++) {
            i32 accumulator = bias_quantized[out_idx];

            for (size_t in_idx = 0; in_idx < totalInputFeatures; in_idx++) {
                // Weight matrix: [input_features, output_features]
                size_t weightIdx = in_idx * outputSize + out_idx;

                // INT8 multiply, accumulate in INT32
                accumulator += static_cast<i32>(input_quantized[in_idx]) *
                               static_cast<i32>(weights_quantized[weightIdx]);
            }

            // Apply zero-point correction: subtract zi * sum_of_weights
            accumulator -= static_cast<i32>(zi) * weight_sum_per_output[out_idx];

            // Dequantize: divide by (Si * Sw)
            fp32 result = static_cast<fp32>(accumulator) / (Si * Sw);

            // Store result in output
            output.get<fp32>(out_idx) = result;
        }

        // Log output statistics
        fp32 output_min = output.get<fp32>(0);
        fp32 output_max = output.get<fp32>(0);
        fp32 output_sum = 0.0f;

        for (size_t i = 0; i < outputSize; i++) {
            fp32 val = output.get<fp32>(i);
            output_min = std::min(output_min, val);
            output_max = std::max(output_max, val);
            output_sum += val;
        }

        //fp32 output_mean = output_sum / outputSize;

        // std::cout << "[QUANT]   Output stats - min: " << output_min
                  //<< ", max: " << output_max
                  //<< ", mean: " << output_mean << std::endl;

        // std::cout << "[QUANT]   Sample FP32 outputs: ";
        for (size_t i = 0; i < std::min(size_t(5), outputSize); i++) {
            std::cout << output.get<fp32>(i) << " ";
        }
        std::cout << std::endl;
    }

}
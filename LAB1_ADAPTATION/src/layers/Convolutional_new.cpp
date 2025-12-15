#include "Convolutional.h"

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
    // INT8 QUANTIZATION - CALIBRATION STATISTICS
    // ==========================================================================

    struct CalibrationStats
    {
        fp32 min, max, mean, std;
        fp32 Si;  // Scale factor
        i8 zi;    // Zero point
    };

    // Global calibration data loaded from JSON
    std::map<std::string, CalibrationStats> calibration_data;
    bool calibration_loaded = false;

    // Helper function to extract layer name from file path (e.g., "conv1_1_weights.bin" -> "conv1_1")
    std::string extractLayerNameFromPath(const std::string& filepath) {
        size_t last_slash = filepath.find_last_of("/\\");
        std::string filename = (last_slash != std::string::npos) ? filepath.substr(last_slash + 1) : filepath;
        
        // Remove "_weights.bin" suffix
        size_t underscore_pos = filename.find("_weights.bin");
        if (underscore_pos != std::string::npos) {
            return filename.substr(0, underscore_pos);
        }
        return filename;
    }

    // Simple JSON parser for calibration stats
    bool loadCalibrationStats(const std::string& json_path)
    {
        if (calibration_loaded) {
            return true;
        }

        std::ifstream file(json_path);
        if (!file.is_open()) {
            logError("Failed to open calibration stats file: " + json_path);
            return false;
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        file.close();

        // Simple JSON parsing - look for layer entries
        size_t pos = 0;
        while ((pos = content.find("\"", pos)) != std::string::npos)
        {
            size_t name_start = pos + 1;
            size_t name_end = content.find("\"", name_start);
            if (name_end == std::string::npos) break;

            std::string layer_name = content.substr(name_start, name_end - name_start);
            pos = name_end + 1;

            // Skip to the opening brace
            size_t brace_start = content.find("{", pos);
            if (brace_start == std::string::npos) break;

            // Find the closing brace
            size_t brace_end = content.find("}", brace_start);
            if (brace_end == std::string::npos) break;

            std::string layer_content = content.substr(brace_start + 1, brace_end - brace_start - 1);

            // Parse the values
            CalibrationStats stats = {};

            // Extract min
            size_t min_pos = layer_content.find("\"min\":");
            if (min_pos != std::string::npos) {
                size_t val_start = layer_content.find(":", min_pos) + 1;
                size_t val_end = layer_content.find(",", val_start);
                if (val_end == std::string::npos) val_end = layer_content.find("}", val_start);
                stats.min = std::stof(layer_content.substr(val_start, val_end - val_start));
            }

            // Extract max
            size_t max_pos = layer_content.find("\"max\":");
            if (max_pos != std::string::npos) {
                size_t val_start = layer_content.find(":", max_pos) + 1;
                size_t val_end = layer_content.find(",", val_start);
                if (val_end == std::string::npos) val_end = layer_content.find("}", val_start);
                stats.max = std::stof(layer_content.substr(val_start, val_end - val_start));
            }

            // Extract mean
            size_t mean_pos = layer_content.find("\"mean\":");
            if (mean_pos != std::string::npos) {
                size_t val_start = layer_content.find(":", mean_pos) + 1;
                size_t val_end = layer_content.find(",", val_start);
                if (val_end == std::string::npos) val_end = layer_content.find("}", val_start);
                stats.mean = std::stof(layer_content.substr(val_start, val_end - val_start));
            }

            // Extract std
            size_t std_pos = layer_content.find("\"std\":");
            if (std_pos != std::string::npos) {
                size_t val_start = layer_content.find(":", std_pos) + 1;
                size_t val_end = layer_content.find(",", val_start);
                if (val_end == std::string::npos) val_end = layer_content.find("}", val_start);
                stats.std = std::stof(layer_content.substr(val_start, val_end - val_start));
            }

            // Extract Si
            size_t Si_pos = layer_content.find("\"Si\":");
            if (Si_pos != std::string::npos) {
                size_t val_start = layer_content.find(":", Si_pos) + 1;
                size_t val_end = layer_content.find(",", val_start);
                if (val_end == std::string::npos) val_end = layer_content.find("}", val_start);
                stats.Si = std::stof(layer_content.substr(val_start, val_end - val_start));
            }

            // Extract zi
            size_t zi_pos = layer_content.find("\"zi\":");
            if (zi_pos != std::string::npos) {
                size_t val_start = layer_content.find(":", zi_pos) + 1;
                size_t val_end = layer_content.find(",", val_start);
                if (val_end == std::string::npos) val_end = layer_content.find("}", val_start);
                stats.zi = static_cast<i8>(std::stoi(layer_content.substr(val_start, val_end - val_start)));
            }

            calibration_data[layer_name] = stats;
            pos = brace_end + 1;
        }

        calibration_loaded = true;
        logInfo("Loaded calibration stats for " + std::to_string(calibration_data.size()) + " layers");

        return true;
    }

    // ==========================================================================
    // Compute the convolution for the layer data
    // Get dimensions from layer parameters
  
    // Perform convolution
    void ConvolutionalLayer::computeNaive(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
        // The following line is an example of copying a single 32-bit floating point integer from the input layer data to the output layer data

        const auto &inputDims = getInputParams().dims;   // [H, W, C_in]
        const auto &outputDims = getOutputParams().dims; // [H_out, W_out, C_out]
        const auto &weightDims = getWeightParams().dims; // [K_H, K_W, C_in, C_out]

        size_t U = 1; // Stride

        size_t W = inputDims[1];
        size_t C = inputDims[2];

        size_t P = outputDims[0];
        size_t Q = outputDims[1];
        size_t M = outputDims[2];

        size_t R = weightDims[0];
        size_t S = weightDims[1];

        #pragma omp parallel for collapse(2)
        for (size_t p = 0; p < P; p++)
        {
            for (size_t q = 0; q < Q; q++)
            {
                for (size_t m = 0; m < M; m++)
                {
                    fp32 result = 0.0f;
                    
                    // Perform the convolution sum
                    // o[p][q][m] = sum_{c,r,s} i[U*p+r][U*q+s][c] * f[r][s][c][m] + b[m]
                    for (size_t c = 0; c < C; c++)
                    { // Input channel
                        for (size_t r = 0; r < R; r++)
                        { // Kernel height
                            for (size_t s = 0; s < S; s++)
                            { // Kernel width
                                // Input coordinates
                                size_t input_h = U * p + r;
                                size_t input_w = U * q + s;
                                
                                // Input index: [input_h, input_w, c]
                                size_t input_idx = input_h * W * C + input_w * C + c;
                                
                                // Weight index: [r, s, c, m]
                                size_t weight_idx = r * S * C * M + s * C * M + c * M + m;
                                
                                // Accumulate
                                result += dataIn.get<fp32>(input_idx) *
                                          getWeightData().get<fp32>(weight_idx);
                            }
                        }
                    }
                    // Add bias: b[m]
                    result += getBiasData().get<fp32>(m);

                    // Output index: [p, q, m]
                    size_t output_idx = p * Q * M + q * M + m;
                    getOutputData().get<fp32>(output_idx) = result;
                }
            }
        }
    }

    // Compute the convolution using threads
    void ConvolutionalLayer::computeThreaded(const LayerData &dataIn) const
    {
        // For simplicity, use naive implementation with thread hints
        computeNaive(dataIn);
    }

    // Compute the convolution using a tiled approach
    void ConvolutionalLayer::computeTiled(const LayerData &dataIn) const
    {
        // For simplicity, use naive implementation
        computeNaive(dataIn);
    }

    // Compute the convolution using SIMD
    void ConvolutionalLayer::computeSIMD(const LayerData &dataIn) const
    {
        // For simplicity, use naive implementation
        computeNaive(dataIn);
    }

    // ==========================================================================
    // INT8 QUANTIZED CONVOLUTION
    // ==========================================================================
    void ConvolutionalLayer::computeQuantized(const LayerData &dataIn) const
    {
        // Load calibration stats if not already loaded
        if (!calibration_loaded) {
            std::vector<std::string> possible_paths = {
                "data/calibration_stats.json",
                "calibration_stats.json",
                "../data/calibration_stats.json"
            };

            bool found = false;
            for (const auto& path : possible_paths) {
                if (loadCalibrationStats(path)) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                logError("Could not find calibration_stats.json file");
                logError("Falling back to FP32 inference");
                computeNaive(dataIn);
                return;
            }
        }

        // Get dimensions
        const auto &inputDims = getInputParams().dims;
        const auto &outputDims = getOutputParams().dims;
        const auto &weightDims = getWeightParams().dims;

        size_t U = 1; // Stride
        size_t W = inputDims[1];
        size_t C = inputDims[2];
        size_t P = outputDims[0];
        size_t Q = outputDims[1];
        size_t M = outputDims[2];
        size_t R = weightDims[0];
        size_t S = weightDims[1];

        // Extract layer name from weight file path to determine calibration stats
        // E.g., "conv1_1_weights.bin" -> "conv1_1"
        std::string layer_name = extractLayerNameFromPath(getWeightParams().filePath);

        // Map layer name to input calibration stats based on architecture
        // Architecture: Conv1_1 -> BN1_1 -> Conv1_2 -> BN1_2 -> Pool1 -> Conv2_1 -> BN2_1 -> Conv2_2 -> BN2_2 -> Pool2 -> Conv3_1 -> BN3_1 -> Conv3_2 -> BN3_2 -> Pool3
        std::string input_stats_name;
        if (layer_name == "conv1_1") {
            input_stats_name = "_input";      // First layer takes raw input
        } else if (layer_name == "conv1_2") {
            input_stats_name = "bn1_1";       // Conv1_2 input comes from BN1_1
        } else if (layer_name == "conv2_1") {
            input_stats_name = "pool1";       // Conv2_1 input comes from Pool1
        } else if (layer_name == "conv2_2") {
            input_stats_name = "bn2_1";       // Conv2_2 input comes from BN2_1
        } else if (layer_name == "conv3_1") {
            input_stats_name = "pool2";       // Conv3_1 input comes from Pool2
        } else if (layer_name == "conv3_2") {
            input_stats_name = "bn3_1";       // Conv3_2 input comes from BN3_1
        } else {
            input_stats_name = "bn3_1";       // Fallback
        }

        auto input_stats_it = calibration_data.find(input_stats_name);
        if (input_stats_it == calibration_data.end()) {
            logError("No calibration stats found for: " + input_stats_name);
            computeNaive(dataIn);
            return;
        }

        const CalibrationStats &input_stats = input_stats_it->second;
        fp32 Si = input_stats.Si;
        i8 zi = input_stats.zi;

        // Log quantization parameters
        std::cout << "[QUANT] " << layer_name << " - Using calibration: " << input_stats_name << std::endl;
        std::cout << "[QUANT]   Input scale (Si): " << Si << ", zero-point (zi): " << static_cast<int>(zi) << std::endl;

        // Calculate weight scale (Sw)
        size_t weight_size = getWeightParams().flat_count();
        fp32 max_weight = 0.0f;
        for (size_t i = 0; i < weight_size; i++) {
            fp32 abs_val = std::abs(getWeightData().get<fp32>(i));
            if (abs_val > max_weight) {
                max_weight = abs_val;
            }
        }
        if (max_weight < 1e-8f) {
            max_weight = 1.0f;
        }
        fp32 Sw = 127.0f / max_weight;

        // Bias scale
        fp32 Sb = Si * Sw;

        std::cout << "[QUANT]   Weight max: " << max_weight << ", scale (Sw): " << Sw << std::endl;
        std::cout << "[QUANT]   Bias scale (Sb): " << Sb << std::endl;

        // Quantize inputs
        size_t input_size = getInputParams().flat_count();
        std::vector<i8> quantized_input(input_size);
        for (size_t i = 0; i < input_size; i++) {
            i32 temp = static_cast<i32>(std::round(Si * dataIn.get<fp32>(i))) + zi;
            quantized_input[i] = static_cast<i8>(std::max<i32>(-128, std::min<i32>(127, temp)));
        }

        // Quantize weights
        std::vector<i8> quantized_weights(weight_size);
        for (size_t i = 0; i < weight_size; i++) {
            i32 temp = static_cast<i32>(std::round(Sw * getWeightData().get<fp32>(i)));
            quantized_weights[i] = static_cast<i8>(std::max<i32>(-128, std::min<i32>(127, temp)));
        }

        // Quantize biases
        std::vector<i32> quantized_biases(M);
        for (size_t m = 0; m < M; m++) {
            quantized_biases[m] = static_cast<i32>(std::round(Sb * getBiasData().get<fp32>(m)));
        }

        // Log sample quantized values (first 5 of each)
        std::cout << "[QUANT]   Sample INT8 inputs: ";
        for (size_t i = 0; i < std::min(size_t(5), input_size); i++) {
            std::cout << static_cast<int>(quantized_input[i]) << " ";
        }
        std::cout << std::endl;

        std::cout << "[QUANT]   Sample INT8 weights: ";
        for (size_t i = 0; i < std::min(size_t(5), weight_size); i++) {
            std::cout << static_cast<int>(quantized_weights[i]) << " ";
        }
        std::cout << std::endl;

        // Precompute sum of quantized weights for each output channel (for zero-point correction)
        std::vector<i32> quantized_weight_sum(M, 0);
        for (size_t m = 0; m < M; m++) {
            for (size_t c = 0; c < C; c++) {
                for (size_t r = 0; r < R; r++) {
                    for (size_t s = 0; s < S; s++) {
                        size_t weight_idx = r * S * C * M + s * C * M + c * M + m;
                        quantized_weight_sum[m] += static_cast<i32>(quantized_weights[weight_idx]);
                    }
                }
            }
        }

        // Main convolution loop (INT8)
        #pragma omp parallel for collapse(2)
        for (size_t p = 0; p < P; p++) {
            for (size_t q = 0; q < Q; q++) {
                for (size_t m = 0; m < M; m++) {
                    i32 accumulator = quantized_biases[m];

                    for (size_t c = 0; c < C; c++) {
                        for (size_t r = 0; r < R; r++) {
                            for (size_t s = 0; s < S; s++) {
                                size_t input_h = U * p + r;
                                size_t input_w = U * q + s;
                                size_t input_idx = input_h * W * C + input_w * C + c;
                                size_t weight_idx = r * S * C * M + s * C * M + c * M + m;

                                // INT8 multiply-accumulate
                                accumulator += static_cast<i32>(quantized_input[input_idx]) *
                                              static_cast<i32>(quantized_weights[weight_idx]);
                            }
                        }
                    }

                    // Apply zero-point correction: subtract zi * sum_of_weights
                    accumulator -= static_cast<i32>(zi) * quantized_weight_sum[m];

                    // Dequantize output
                    fp32 result = static_cast<fp32>(accumulator) / (Si * Sw);

                    // NO ReLU here - it's fused with BatchNorm in our architecture

                    size_t output_idx = p * Q * M + q * M + m;
                    getOutputData().get<fp32>(output_idx) = result;
                }
            }
        }

        // Log output statistics
        size_t output_size = P * Q * M;
        fp32 output_min = getOutputData().get<fp32>(0);
        fp32 output_max = getOutputData().get<fp32>(0);
        fp32 output_sum = 0.0f;

        for (size_t i = 0; i < output_size; i++) {
            fp32 val = getOutputData().get<fp32>(i);
            output_min = std::min(output_min, val);
            output_max = std::max(output_max, val);
            output_sum += val;
        }

        fp32 output_mean = output_sum / output_size;

        std::cout << "[QUANT]   Output stats - min: " << output_min
                  << ", max: " << output_max
                  << ", mean: " << output_mean << std::endl;

        std::cout << "[QUANT]   Sample FP32 outputs: ";
        for (size_t i = 0; i < std::min(size_t(5), output_size); i++) {
            std::cout << getOutputData().get<fp32>(i) << " ";
        }
        std::cout << std::endl;
    }

} // namespace ML

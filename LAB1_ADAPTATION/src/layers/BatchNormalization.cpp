#include "BatchNormalization.h"
#include <cmath>
#include <iostream>

namespace ML {

BatchNormalizationLayer::BatchNormalizationLayer(
    const LayerParams& inputParams,
    const LayerParams& outputParams,
    const LayerParams& meanParams,
    const LayerParams& varianceParams,
    const LayerParams& gammaParams,
    const LayerParams& betaParams,
    float epsilon
) : Layer(inputParams, outputParams, Layer::LayerType::BATCHNORM),
    meanParam(meanParams),
    varianceParam(varianceParams),
    gammaParam(gammaParams),
    betaParam(betaParams),
    mean(meanParams),
    variance(varianceParams),
    gamma(gammaParams),
    beta(betaParams),
    epsilon(epsilon),
    std_dev(meanParams),  // Same shape as mean
    std_dev_computed(false)
{
    // Verify parameter shapes match
    if (meanParam.flat_count() != varianceParam.flat_count() ||
        meanParam.flat_count() != gammaParam.flat_count() ||
        meanParam.flat_count() != betaParam.flat_count()) {
        throw std::runtime_error("BatchNormalization: Parameter shapes must match");
    }
}

void BatchNormalizationLayer::allocLayer() {
    Layer::allocLayer();

    // Load parameters from files
    mean.allocData();
    mean.loadData();

    variance.allocData();
    variance.loadData();

    gamma.allocData();
    gamma.loadData();

    beta.allocData();
    beta.loadData();

    // Pre-allocate std_dev
    std_dev.allocData();
}

void BatchNormalizationLayer::freeLayer() {
    Layer::freeLayer();
    mean.freeData();
    variance.freeData();
    gamma.freeData();
    beta.freeData();
    std_dev.freeData();
}

void BatchNormalizationLayer::computeStdDev() const {
    if (std_dev_computed) return;

    const size_t numChannels = mean.getParams().flat_count();
    for (size_t i = 0; i < numChannels; i++) {
        float var = variance.get<fp32>(i);
        std_dev.get<fp32>(i) = std::sqrt(var + epsilon);
    }
    std_dev_computed = true;
}

void BatchNormalizationLayer::computeNaive(const LayerData& input) const {
    // Pre-compute std_dev if not already done
    computeStdDev();

    LayerData& output = getOutputData();
    const auto& inputDims = input.getParams().dims;
    const size_t numChannels = mean.getParams().flat_count();

    // Determine input format
    if (inputDims.size() == 4) {
        // 4D input: (batch, height, width, channels) - Conv layer output
        const size_t batch = inputDims[0];
        const size_t height = inputDims[1];
        const size_t width = inputDims[2];
        const size_t channels = inputDims[3];

        if (channels != numChannels) {
            throw std::runtime_error("BatchNormalization: Channel count mismatch");
        }

        for (size_t b = 0; b < batch; b++) {
            for (size_t h = 0; h < height; h++) {
                for (size_t w = 0; w < width; w++) {
                    for (size_t c = 0; c < channels; c++) {
                        size_t idx = ((b * height + h) * width + w) * channels + c;

                        float x = input.get<fp32>(idx);
                        float mean_c = mean.get<fp32>(c);
                        float std_c = std_dev.get<fp32>(c);
                        float gamma_c = gamma.get<fp32>(c);
                        float beta_c = beta.get<fp32>(c);

                        // y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
                        float normalized = (x - mean_c) / std_c;
                        float y = gamma_c * normalized + beta_c;

                        // Apply ReLU activation
                        y = std::max(0.0f, y);

                        output.get<fp32>(idx) = y;
                    }
                }
            }
        }
    }
    else if (inputDims.size() == 3) {
        // 3D input: (height, width, channels) - Single sample
        const size_t height = inputDims[0];
        const size_t width = inputDims[1];
        const size_t channels = inputDims[2];

        if (channels != numChannels) {
            throw std::runtime_error("BatchNormalization: Channel count mismatch");
        }

        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                for (size_t c = 0; c < channels; c++) {
                    size_t idx = (h * width + w) * channels + c;

                    float x = input.get<fp32>(idx);
                    float mean_c = mean.get<fp32>(c);
                    float std_c = std_dev.get<fp32>(c);
                    float gamma_c = gamma.get<fp32>(c);
                    float beta_c = beta.get<fp32>(c);

                    // y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
                    float normalized = (x - mean_c) / std_c;
                    float y = gamma_c * normalized + beta_c;

                    // Apply ReLU activation
                    y = std::max(0.0f, y);

                    output.get<fp32>(idx) = y;
                }
            }
        }
    }
    else if (inputDims.size() == 1) {
        // 1D input: (features) - Dense layer output
        const size_t features = inputDims[0];

        if (features != numChannels) {
            throw std::runtime_error("BatchNormalization: Feature count mismatch");
        }

        for (size_t i = 0; i < features; i++) {
            float x = input.get<fp32>(i);
            float mean_i = mean.get<fp32>(i);
            float std_i = std_dev.get<fp32>(i);
            float gamma_i = gamma.get<fp32>(i);
            float beta_i = beta.get<fp32>(i);

            // y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
            float normalized = (x - mean_i) / std_i;
            float y = gamma_i * normalized + beta_i;

            // Apply ReLU activation
            y = std::max(0.0f, y);

            output.get<fp32>(i) = y;
        }
    }
    else {
        throw std::runtime_error("BatchNormalization: Unsupported input dimensionality");
    }
}

void BatchNormalizationLayer::computeThreaded(const LayerData& input) const {
    // For now, fall back to naive implementation
    // TODO: Implement multi-threaded version
    computeNaive(input);
}

void BatchNormalizationLayer::computeTiled(const LayerData& input) const {
    // For now, fall back to naive implementation
    // TODO: Implement cache-tiled version
    computeNaive(input);
}

void BatchNormalizationLayer::computeSIMD(const LayerData& input) const {
    // For now, fall back to naive implementation
    // TODO: Implement SIMD vectorized version
    computeNaive(input);
}

} // namespace ML

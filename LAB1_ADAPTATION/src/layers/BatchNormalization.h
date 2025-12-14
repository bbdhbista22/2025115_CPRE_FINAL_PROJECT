#pragma once

#include "Layer.h"
#include <cmath>

namespace ML {

/**
 * BatchNormalization Layer
 *
 * Implements batch normalization for inference mode:
 * y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
 *
 * Parameters:
 * - mean: Running mean from training (per channel)
 * - variance: Running variance from training (per channel)
 * - gamma: Scale parameter (per channel)
 * - beta: Shift parameter (per channel)
 * - epsilon: Small constant for numerical stability (default: 1e-3 for Keras)
 */
class BatchNormalizationLayer : public Layer {
public:
    BatchNormalizationLayer(
        const LayerParams& inputParams,
        const LayerParams& outputParams,
        const LayerParams& meanParams,
        const LayerParams& varianceParams,
        const LayerParams& gammaParams,
        const LayerParams& betaParams,
        float epsilon = 1e-3f  // Keras default epsilon
    );

    ~BatchNormalizationLayer() override = default;

    void allocLayer() override;
    void freeLayer() override;

    void computeNaive(const LayerData& dataIn) const override;
    void computeThreaded(const LayerData& dataIn) const override;
    void computeTiled(const LayerData& dataIn) const override;
    void computeSIMD(const LayerData& dataIn) const override;
    void computeQuantized(const LayerData& dataIn) const override;

private:
    LayerParams meanParam;
    LayerParams varianceParam;
    LayerParams gammaParam;
    LayerParams betaParam;

    mutable LayerData mean;
    mutable LayerData variance;
    mutable LayerData gamma;
    mutable LayerData beta;
    float epsilon;

    // Cached sqrt(variance + epsilon) for efficiency
    mutable LayerData std_dev;
    mutable bool std_dev_computed;

    void computeStdDev() const;
};

} // namespace ML

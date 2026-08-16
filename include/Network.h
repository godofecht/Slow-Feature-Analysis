#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

class Network
{
public:
    Network(const std::vector<unsigned> &topology);
    void backPropagate(const std::vector<double> &targetVals);
    void feedForward(std::vector<double> &inputVals);
    void getResults(std::vector<double> &resultVals);
    double getRecentAverageError(void) const { return m_recentAverageError; }

    std::vector<double> GetWeights() const;
    void PutWeights(std::vector<double> &weights);

    void UpdateWeights();
    void NormalizeWeights(int connection_index);

    std::vector<unsigned> GetTopology() const { return m_topology; }

    inline void setDeltaWeight(int layer, int neuron, int output_neuron, double dw) {
        m_deltaWeights[layer_weight_offsets[layer] + neuron * m_topology[layer + 1] + output_neuron] = dw;
    }

    inline double getWeight(int layer, int neuron, int output_neuron) const {
        return m_weights[layer_weight_offsets[layer] + neuron * m_topology[layer + 1] + output_neuron];
    }

    inline void setOutputVal(int layer, int neuron, double val) {
        m_outputs[layer_offsets[layer] + neuron] = val;
    }

    inline double getOutputVal(int layer, int neuron) const {
        return m_outputs[layer_offsets[layer] + neuron];
    }

    std::vector<double> m_weights;
    std::vector<double> m_deltaWeights;
    std::vector<double> m_outputs;

    std::vector<unsigned> m_topology;
    std::vector<unsigned> layer_offsets;
    std::vector<unsigned> layer_weight_offsets;

private:
    double m_gradient;
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
};

#endif

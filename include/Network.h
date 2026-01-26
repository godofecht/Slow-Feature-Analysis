#ifndef NETWORK_H
#define NETWORK_H

#include "NN.h"
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

    std::vector<Layer> GetLayers()
    {
        return m_layers;
    }

    std::vector<Layer> m_layers;

private:
    double m_gradient;
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;
};

#endif

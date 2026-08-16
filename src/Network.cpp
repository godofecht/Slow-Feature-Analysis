#include "Network.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cassert>
#include <iostream>

using namespace std;

Network::Network(const vector<unsigned> &topology)
{
    srand((unsigned int)time(NULL));

    m_topology = topology;
    unsigned numLayers = topology.size();

    unsigned total_neurons = 0;
    unsigned total_weights = 0;

    for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
        layer_offsets.push_back(total_neurons);
        layer_weight_offsets.push_back(total_weights);

        unsigned numNeurons = topology[layerNum];
        total_neurons += numNeurons;

        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
        total_weights += numNeurons * numOutputs;
    }

    m_outputs.assign(total_neurons, 0.0);
    m_weights.assign(total_weights, 0.0);
    m_deltaWeights.assign(total_weights, 0.0);

    for (size_t i = 0; i < m_weights.size(); ++i) {
        m_weights[i] = ((rand() / double(RAND_MAX)) * 2.0 - 1.0);
    }

    // for stone and bray sfa
    for (int i = 0; i < (int)topology.back(); i++)
    {
        NormalizeWeights(i);
    }
}

void Network::NormalizeWeights(int connection_index)
{
    double sum_weights_squared = 0.0;

    for (unsigned layerNum = 1; layerNum < m_topology.size(); layerNum++) {
        unsigned prevLayerSize = m_topology[layerNum - 1];
        unsigned weightOffset = layer_weight_offsets[layerNum - 1];
        unsigned numOutputs = m_topology[layerNum];

        #pragma omp simd reduction(+:sum_weights_squared)
        for (unsigned n = 0; n < prevLayerSize; n++) {
            sum_weights_squared += m_weights[weightOffset + n * numOutputs + connection_index];
        }
    }

    double average = sum_weights_squared / 101.0; // TODO: optimization - remove hardcoded 101.0
    sum_weights_squared = 0.0;

    for (unsigned layerNum = 1; layerNum < m_topology.size(); layerNum++) {
        unsigned prevLayerSize = m_topology[layerNum - 1];
        unsigned weightOffset = layer_weight_offsets[layerNum - 1];
        unsigned numOutputs = m_topology[layerNum];

        #pragma omp simd reduction(+:sum_weights_squared)
        for (unsigned n = 0; n < prevLayerSize; n++) {
            double w = m_weights[weightOffset + n * numOutputs + connection_index] - average;
            m_weights[weightOffset + n * numOutputs + connection_index] = w;
            sum_weights_squared += w * w;
        }
    }

    double checksum = 0.0;
    for (unsigned layerNum = 1; layerNum < m_topology.size(); layerNum++) {
        unsigned prevLayerSize = m_topology[layerNum - 1];
        unsigned weightOffset = layer_weight_offsets[layerNum - 1];
        unsigned numOutputs = m_topology[layerNum];

        #pragma omp simd reduction(+:checksum)
        for (unsigned n = 0; n < prevLayerSize; n++) {
            double newWeight = m_weights[weightOffset + n * numOutputs + connection_index] / sqrt(sum_weights_squared);
            m_weights[weightOffset + n * numOutputs + connection_index] = newWeight;
            checksum += newWeight * newWeight;
        }
    }
    cout << checksum << endl;
}

void Network::UpdateWeights()
{
    double* weights_ptr = m_weights.data();
    const double* delta_weights_ptr = m_deltaWeights.data();
    size_t size = m_weights.size();

    #pragma omp target teams distribute parallel for map(tofrom: weights_ptr[0:size]) map(to: delta_weights_ptr[0:size])
    for (size_t i = 0; i < size; ++i) {
        weights_ptr[i] += delta_weights_ptr[i];
    }
}

void Network::feedForward(vector<double> &inputVals)
{
    assert(inputVals.size() == m_topology[0]);
    // Assign (latch) the input values into the input neurons
    for (unsigned i = 0; i < inputVals.size(); i++) {
        m_outputs[i] = inputVals[i];
    }

    // Get raw pointers for GPU offload
    double* outputs_ptr = m_outputs.data();
    const double* weights_ptr = m_weights.data();

    // forward propagate
    for (unsigned layerNum = 1; layerNum < m_topology.size(); ++layerNum) {
        unsigned prevLayerSize = m_topology[layerNum - 1];
        unsigned prevLayerOffset = layer_offsets[layerNum - 1];
        unsigned weightOffset = layer_weight_offsets[layerNum - 1];

        unsigned currentLayerSize = m_topology[layerNum];
        unsigned currentLayerOffset = layer_offsets[layerNum];

        #pragma omp target teams distribute parallel for map(to: weights_ptr[weightOffset:prevLayerSize*currentLayerSize]) map(tofrom: outputs_ptr[0:m_outputs.size()])
        for (unsigned n = 0; n < currentLayerSize; n++) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (unsigned p = 0; p < prevLayerSize; p++) {
                sum += outputs_ptr[prevLayerOffset + p] *
                       weights_ptr[weightOffset + p * currentLayerSize + n];
            }
            outputs_ptr[currentLayerOffset + n] = sum;
        }
    }
}

void Network::getResults(vector<double> &resultVals)
{
    resultVals.clear();
    unsigned lastLayerOffset = layer_offsets.back();
    unsigned lastLayerSize = m_topology.back();
    for (unsigned n = 0; n < lastLayerSize; n++) {
        resultVals.push_back(m_outputs[lastLayerOffset + n]);
    }
}

vector<double> Network::GetWeights() const
{
    return m_weights;
}

void Network::PutWeights(vector<double> &weights)
{
    m_weights = weights;
}

void Network::backPropagate(const vector<double> &targetVals)
{
    // Not implemented in the original code but declared in the header.
}

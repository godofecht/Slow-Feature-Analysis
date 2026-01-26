#include "NN.h"
#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <ctime>
#include <cstdlib>

using namespace std;

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c) {
        m_outputWeights.push_back(connection());
        m_outputWeights[c].weight = ((rand() / double(RAND_MAX)) * 2.0f - 1.0f);
        m_outputWeights[c].deltaweight = 0.0f;
    }
    m_myIndex = myIndex;
}

void Neuron::feedForward(Layer &prevLayer)
{
    double sum = 0.0;
    for (unsigned n = 0; n < prevLayer.size(); ++n) {
        sum += prevLayer[n].getOutputVal() *
               prevLayer[n].m_outputWeights[m_myIndex].weight;
    }
    m_outputVal = sum;
}

double Neuron::getOutputVal() const
{
    return m_outputVal;
}

void Neuron::setOutputVal(double n)
{
    m_outputVal = n;
}

int Neuron::getIndex() const
{
    return m_myIndex;
}

void Neuron::NaivelyUpdateWeights()
{
    for (size_t i = 0; i < m_outputWeights.size(); i++)
    {
        double oldDeltaWeight = m_outputWeights[i].deltaweight;
        m_outputWeights[i].weight += oldDeltaWeight;
    }
}

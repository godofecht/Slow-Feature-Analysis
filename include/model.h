#ifndef MODEL_H
#define MODEL_H

#include "Network.h"
#include <vector>
#include <iostream>

class Model
{
public:
    Network *thisNetwork;
    std::vector<unsigned> topology;
    std::vector<double> weights;

public:
    Model();
    virtual ~Model();

    void SetTopology(const std::vector<unsigned> &tp);
    void InitializeTopology();
    void BackPropagate(const std::vector<double> &targetVals);
    Network *getNetwork();
    std::vector<double> GetWeights();
    void feedforward(std::vector<double> &inputs);
    std::vector<double> GetResult();
    void SetWeights(std::vector<double> &weights);
    void DisplayTopology();
    void UpdateWeights();
};

#endif

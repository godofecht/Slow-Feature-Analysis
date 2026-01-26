#include "model.h"

using namespace std;

Model::Model() : thisNetwork(nullptr)
{
}

Model::~Model()
{
    delete thisNetwork;
}

void Model::SetTopology(const vector<unsigned> &tp)
{
    topology = tp;
}

void Model::InitializeTopology()
{
    if (thisNetwork) delete thisNetwork;
    thisNetwork = new Network(topology);
}

void Model::BackPropagate(const vector<double> &targetVals)
{
    if (thisNetwork) thisNetwork->backPropagate(targetVals);
}

Network *Model::getNetwork()
{
    return thisNetwork;
}

vector<double> Model::GetWeights()
{
    if (thisNetwork) weights = thisNetwork->GetWeights();
    return weights;
}

void Model::feedforward(vector<double> &inputs)
{
    if (thisNetwork) thisNetwork->feedForward(inputs);
}

vector<double> Model::GetResult()
{
    vector<double> resultVals;
    if (thisNetwork) thisNetwork->getResults(resultVals);
    return resultVals;
}

void Model::SetWeights(vector<double> &weights)
{
    if (thisNetwork) thisNetwork->PutWeights(weights);
}

void Model::DisplayTopology()
{
    for (unsigned int i : topology)
    {
        cout << i << "\n";
    }
}

void Model::UpdateWeights()
{
    if (thisNetwork) thisNetwork->UpdateWeights();
}

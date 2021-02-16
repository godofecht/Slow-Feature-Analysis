#ifndef MODEL_H
#define MODEL_H

#include "NN.h"
#include "Network.h"
#include "Network.cpp"
#include <iostream>

class Model
{
public:
	Network* thisNetwork;
	vector<unsigned> topology;
	vector<double>weights;
	public:

	Model()
	{
	}
    
    void SetTopology(int* tp)
	{
        int arrSize = *(&tp + 1) - tp;
        vector<unsigned> topo;
        for(int i=0;i<arrSize;i++)
        {
            topo.push_back(tp[i]);
        }
		topology = topo;
	} 

    void SetTopology(vector<unsigned> tp)
    {
        topology = tp;
    }
	void InitializeTopology()
	{
		thisNetwork = new Network(topology);
	}
	void BackPropagate(vector<double> targetVals)
	{
		thisNetwork->backPropagate(targetVals);
	}
	Network* getNetwork()
	{
		return thisNetwork;
	}
	vector<double> GetWeights()
	{
		weights = thisNetwork->GetWeights();
		return weights;
	}
	void feedforward(vector<double> inputs)
	{
		thisNetwork->feedForward(inputs);
	}
	vector<double> GetResult()
	{
		vector<double> resultVals;
		thisNetwork->getResults(resultVals);
		return resultVals;
	}
	void SetWeights(vector<double> weights)
	{
		thisNetwork->PutWeights(weights);
	}

    void DisplayTopology()
    {
        for(int i=0;i<topology.size();i++)
        {
            std::cout<<topology[i]<<"\n";
        }
    }
	
	void UpdateWeights()
	{
	//	cout<<"ok";
		thisNetwork->UpdateWeights();
	}
};
#endif
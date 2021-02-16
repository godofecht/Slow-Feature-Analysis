
#include "Network.h"
#include "NN.cpp"
#include <vector>

Network::Network(const vector <unsigned> &topology)
{
	srand((unsigned int)time(NULL));

	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; layerNum++) {
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];



		// We have a new layer, now fill it with neurons, and a bias neuron in each layer.
		for (unsigned neuronNum = 0; neuronNum < topology[layerNum]; neuronNum++) {
			m_layers.back().push_back(Neuron(numOutputs, neuronNum)); 
		}
	}


	NormalizeWeights(0);
	NormalizeWeights(1); //because we have two output neurons
}

void Network::NormalizeWeights(int connection_index)
{
	double sum_weights_squared = 0.0f;
	double checksum = 0.0f;
	for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {
		Layer& layer = m_layers[layerNum];
		Layer& prevLayer = m_layers[layerNum - 1];




		for (unsigned n = 0; n < prevLayer.size(); n++) {
			Neuron* neuron = &(prevLayer[n]);

			sum_weights_squared = sum_weights_squared +neuron->m_outputWeights[connection_index].weight;// pow(neuron->m_outputWeights[0].weight,2);
		}
	}
	double average = sum_weights_squared/101.0f;
	sum_weights_squared = 0.0f;
	for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {
		Layer& layer = m_layers[layerNum];
		Layer& prevLayer = m_layers[layerNum - 1];




		for (unsigned n = 0; n < prevLayer.size(); n++) {
			Neuron* neuron = &(prevLayer[n]);
			neuron->m_outputWeights[connection_index].weight -= average;
			sum_weights_squared += pow(neuron->m_outputWeights[connection_index].weight,2);
			// pow(neuron->m_outputWeights[0].weight,2);
		}
	}

	for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {
		Layer& layer = m_layers[layerNum];
		Layer& prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < prevLayer.size(); n++) {
			Neuron* neuron = &(prevLayer[n]);
			double weight_squared = pow(neuron->m_outputWeights[connection_index].weight,2);
			double newWeight = neuron->m_outputWeights[connection_index].weight/sqrt(sum_weights_squared);
			neuron->m_outputWeights[connection_index].weight = newWeight;
		//	cout<<newWeight<<endl;
	//			cout<<neuron->m_outputWeights[0].weight<<endl;
			checksum+= pow(newWeight,2);
		}

	}
	cout<<checksum<<endl;
	double checksumfactor = 1.0f/checksum;
	checksum = 0.0f;

	/*
	for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {
		Layer& layer = m_layers[layerNum];
		Layer& prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < prevLayer.size(); n++) {
			Neuron* neuron = &(prevLayer[n]);
			double weight_squared = pow(neuron->m_outputWeights[0].weight,2);
			double newWeight =(neuron->m_outputWeights[0].weight*sqrt(checksumfactor));// sqrt(1.0f-sum_weights_squared + weight_squared);
			neuron->m_outputWeights[0].weight = newWeight;
		//	cout<<newWeight<<endl;
	//			cout<<neuron->m_outputWeights[0].weight<<endl;
			checksum+= pow(newWeight,2);
		}
		double checksumfactor = 1.0f/ checksum;
		cout<<checksum<<endl;
	}
	*/
}

void Network::UpdateWeights()
{
	for (unsigned layerNum = 1; layerNum < m_layers.size(); layerNum++) {
		Layer& layer = m_layers[layerNum];
		Layer& prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < prevLayer.size(); n++) {
			Neuron* neuron = &(prevLayer[n]);

			neuron->NaivelyUpdateWeights();
		}
	}	
}





void Network::feedForward(vector <double> &inputVals)
{
	assert(inputVals.size() == m_layers[0].size());
	// Assign (latch) the input values into the input neurons
	for (unsigned i = 0; i < inputVals.size(); i++) {
		m_layers[0][i].setOutputVal(inputVals[i]);
	}
	// forward propagate
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
		Layer &prevLayer = m_layers[layerNum - 1];
		for (unsigned n = 0; n < m_layers[layerNum].size(); n++) {
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}

void  Network::getResults(vector <double> &resultVals)
{
	resultVals.clear();
	for (unsigned n = 0; n < m_layers.back().size() ; n++) {
		resultVals.push_back(m_layers.back()[n].getOutputVal());
	}
}

vector<double>  Network::GetWeights() const
{
	//this will hold the weights
	vector<double> weights;
	//for each layer
	for (int i = 0; i<m_layers.size()-1; ++i)
	{
		//for each neuron
		for (int j = 0; j<m_layers[i].size(); ++j)
		{
			//for each weight
			for (int k = 0; k<m_layers[i][j].m_outputWeights.size(); ++k)
			{
				weights.push_back(m_layers[i][j].m_outputWeights[k].weight);
			}
		}
	}
	return weights;
}

void Network::PutWeights(vector<double> &weights)
{
	int cWeight = 0;
	//for each layer
	for (int i = 0; i<m_layers.size()-1; ++i)
	{
		//for each neuron
		for (int j = 0; j<m_layers[i].size(); ++j)
		{
			//for each weight
			for (int k = 0; k<m_layers[i][j].m_outputWeights.size(); ++k)
			{
				m_layers[i][j].m_outputWeights[k].weight = weights[cWeight++];
			}
		}
	}
}
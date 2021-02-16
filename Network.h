#ifndef NETWORK_H
#define NETWORK_H

#include "NN.h"

class Network
{
	public:
		Network(const vector <unsigned> &topology);
		void backPropagate(const vector <double> &targetVals);
		void feedForward(vector <double> &inputVals);
		void getResults(vector <double> &resultVals);
		double getRecentAverageError(void) const { return m_recentAverageError; }

		vector<double> GetWeights() const;
		void PutWeights(vector<double> &weights);

	    void UpdateWeights();

		void NormalizeWeights(int connection_index);

		vector<Layer> GetLayers()
		{
 			return m_layers;
		}


		vector <Layer> m_layers;
	private:

		double m_gradient;
		double m_error;
		double m_recentAverageError;
		double m_recentAverageSmoothingFactor;
};

#endif
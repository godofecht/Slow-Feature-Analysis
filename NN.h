#ifndef NN_H
#define NN_H

#include<vector>
#include<cmath>
#include<cassert>
#include<iostream>

using namespace std;

class Neuron;
typedef vector<Neuron> Layer;
class connection
{
public:
	double weight;
	double deltaweight;

	void setDW(double dw)
	{
		deltaweight = dw;
	}
};
class Neuron
{
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void feedForward(Layer &prevLayer);
	double getOutputVal();
	void setOutputVal(double n);
	void calcHiddenGradients(const Layer &nextLayer);
	void calcOutputGradients(double targetVal);
	void updateInputWeights(Layer &prevLayer);
	void NaivelyUpdateWeights();
    double eta;
    double alpha;
	vector <connection> m_outputWeights;

	int getIndex();
private:
	double m_gradient;
	double m_outputVal;
	static double randomWeight();
	unsigned m_myIndex;
	double sumDOW(const Layer &nextLayer) const;
	static double transferFunctionDerivative(double x);
	static double transferFunction(double x);
};


#endif
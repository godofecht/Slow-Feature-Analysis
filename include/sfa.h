#ifndef SFA_H
#define SFA_H

#include "model.h"
#include <string>
#include <vector>
#include <utility>

class SFA : public Model
{
    std::vector<double> corVector1, corVector2;
    double alpha = 0.001;

    double gamma_long = 0.0, gamma_short = 0.0;
    double lambda_long = 0.0, lambda_short = 0.0;

    std::vector<double> x_tilde_vector;
    std::vector<double> x_bar_vector;
    std::vector<double> del_weight_vector;

    std::vector<double> del_weight1_vector;
    std::vector<double> del_weight2_vector;

    std::vector<double> input_vector;

    std::vector<double> y1_vector;
    std::vector<double> y2_vector;

    double y_tilde = 0.0, y_bar = 0.0, U = 0.0, V = 0.0;

    double y1_tilde = 0.0, y2_tilde = 0.0, y1_bar = 0.0, y2_bar = 0.0, U1 = 0.0, U2 = 0.0, V1 = 0.0, V2 = 0.0;
    double y1 = 0.0, y2 = 0.0;

    unsigned NUM_INPUT_NEURONS_X = 0;
    unsigned NUM_INPUT_NEURONS_Y = 0;
    unsigned TIMES_TO_RUN = 1;
    double ro = 450.0;
    int TOTAL_TIME = 0;
    std::vector<double> signal;
    std::vector<double> resultVector;
    std::vector<double> resultVector1;
    std::vector<double> resultVector2;

    std::vector<double> inputVector;
    std::vector<double> resultUni;
    std::vector<double> resultTuple;
    bool bAdapt = false;
    bool bForward = true;
    double y = 0.0;

    int NUM_INVARIANCES = 0;

    std::vector<double> signalVector;
    std::vector<double> signalVector1;
    std::vector<double> signalVector2;

    std::vector<double> f_vector1;
    std::vector<double> f_vector2;

public:
    SFA(float RO, int num_invariances);
    virtual ~SFA() {}

    double GetCorrelationTrace(const std::vector<double> &va, const std::vector<double> &vb, int t);
    void reset1Inv();
    void write_csv(std::string filename, std::vector<std::pair<std::string, std::vector<double>>> dataset);
    void Train();
    void TrainTwoInvariances();

    int GetSignalValue(int time);
    std::pair<int, int> GetSignalTuple(int time);

    double CalculateDelWeights(int i);
    double CalculateDelWeight(double v, double u, double output, double y_b, double y_ti, int i);

    double getYBar(double yz, double y_barz);
    double getYTilde(double yz, double y_tildez);
    double getXBar(double x_barz, double x);
    double getXTilde(double x_tildez, double x);
    double getV(double Vz, double Y, double Y_BAR);
    double getU(double Uz, double Y, double Y_TILDE);

    double GetOutput(int signal_value);
    void UpdateX(int i);
    double GetY(int sig_val);
    std::pair<double, double> GetYTuple(int val1, int val2, int time_step);
    double GetWeightedAntiHebbian(int time_step);
    std::pair<double, double> GetOutputTuple(int val1, int val2);

    void UpdateNeuron(Neuron *neuron, int neuron_index);
    void UpdateNeuronWithDelta(Neuron *neuron, double dw, int output_neuron_index);

    void OscillateFeedForward(int signal_value, int time_step);
    void OscillateFeedForwardTuple(int signal_value1, int signal_value2, int time_step);

    std::vector<double> GenerateInputs(int number_to_encode);
    std::vector<double> GenerateInputsFromTuple(int number1, int number2);
};

#endif

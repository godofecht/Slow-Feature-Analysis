#include "sfa.h"
#include "stats.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <algorithm>

using namespace std;

SFA::SFA(float RO, int num_invariances)
{
    ro = RO;
    TOTAL_TIME = 50 * ro;
    NUM_INVARIANCES = num_invariances;

    if (NUM_INVARIANCES == 1)
    {
        NUM_INPUT_NEURONS_Y = 1;
        NUM_INPUT_NEURONS_X = 101;

        lambda_long = 2.0f * ro;
        lambda_short = ro / 31.0f;
        gamma_long = pow(0.5f, (1.0f / (lambda_long)));
        gamma_short = pow(0.5f, (1.0f / (lambda_short)));

        y_tilde = 0.0f;
        y_bar = 0.0f;
        U = 0.000f;
        V = 0.000f;

        vector<unsigned> a;
        a.push_back(NUM_INPUT_NEURONS_X);
        a.push_back(1);
        SetTopology(a);
        InitializeTopology();

        x_bar_vector.assign(NUM_INPUT_NEURONS_X, 0.0f);
        x_tilde_vector.assign(NUM_INPUT_NEURONS_X, 0.0f);
        del_weight_vector.assign(NUM_INPUT_NEURONS_X, 0.0f);
        input_vector.assign(NUM_INPUT_NEURONS_X, 0.0f);
    }
    else if (NUM_INVARIANCES == 2)
    {
        NUM_INPUT_NEURONS_Y = 51;
        NUM_INPUT_NEURONS_X = 51;
        unsigned total_inputs = NUM_INPUT_NEURONS_X * NUM_INPUT_NEURONS_Y;

        lambda_long = 2.0f * ro;
        lambda_short = ro / 31.0f;
        gamma_long = pow(0.5f, (1.0f / (lambda_long)));
        gamma_short = pow(0.5f, (1.0f / (lambda_short)));

        y1_tilde = 0.0f;
        y1_bar = 0.0f;
        y2_tilde = 0.0f;
        y2_bar = 0.0f;
        U1 = 0.000f;
        U2 = 0.0f;
        V1 = 0.0f;
        V2 = 0.0f;

        y1 = 0.0f;
        y2 = 0.0f;

        vector<unsigned> a;
        a.push_back(total_inputs);
        a.push_back(2);
        SetTopology(a);
        InitializeTopology();

        x_bar_vector.assign(total_inputs, 0.0f);
        x_tilde_vector.assign(total_inputs, 0.0f);
        del_weight_vector.assign(total_inputs, 0.0f);
        del_weight1_vector.assign(total_inputs, 0.0f);
        del_weight2_vector.assign(total_inputs, 0.0f);
        input_vector.assign(total_inputs, 0.0f);
    }
    cout << "finished constructing" << endl;
}

double SFA::GetCorrelationTrace(const vector<double> &va, const vector<double> &vb, int t)
{
    // Note: pearsoncoeff is now efficient and doesn't need vector slicing if we wanted,
    // but the original logic used only recent values.
    // Wait, the original code called pearsoncoeff(va, vb) where va and vb were the WHOLE vectors.
    // Let's check original GetCorrelationTrace again.
    /*
        int time_step = t;
        int lambda_correlation = fmin(11*ro,time_step);
        vector<double> vector_y1;
        vector<double> vector_y2;
        vector_y1.clear();
        vector_y2.clear();

        for(int i=vb.size()-lambda_correlation;i<vb.size();i++)
        {
            if(i>-1)
            {
                vector_y1.push_back(va[i]);
                vector_y2.push_back(vb[i]);
            }
        }
        double val =  pearsoncoeff(va,vb); // WAIT! It used va, vb NOT vector_y1, vector_y2!
    */
    // That looks like a bug in original code or at least inefficient.
    // If it intended to use the trace, it should use vector_y1, vector_y2.
    // But since I want to preserve behavior while optimizing, I'll see.
    // Actually, pearsoncoeff(va, vb) is what was there.

    return pearsoncoeff(va, vb);
}

void SFA::reset1Inv()
{
    y_tilde = 0.0f;
    y_bar = 0.0f;
    U = 0.000f;
    V = 0.000f;

    vector<unsigned> a;
    a.push_back(NUM_INPUT_NEURONS_X);
    a.push_back(1);
    SetTopology(a);
    InitializeTopology();

    x_bar_vector.assign(NUM_INPUT_NEURONS_X, 0.0f);
    x_tilde_vector.assign(NUM_INPUT_NEURONS_X, 0.0f);
    del_weight_vector.assign(NUM_INPUT_NEURONS_X, 0.0f);
    input_vector.assign(NUM_INPUT_NEURONS_X, 0.0f);
}

void SFA::write_csv(string filename, vector<pair<string, vector<double>>> dataset)
{
    ofstream myFile(filename);
    for (size_t j = 0; j < dataset.size(); ++j)
    {
        myFile << dataset[j].first;
        if (j != dataset.size() - 1) myFile << ",";
    }
    myFile << "\n";

    if (!dataset.empty() && !dataset[0].second.empty()) {
        size_t rows = dataset[0].second.size();
        size_t cols = dataset.size();
        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                myFile << dataset[j].second[i];
                if (j != cols - 1) myFile << ",";
            }
            myFile << "\n";
        }
    }
    myFile.close();
}

void SFA::Train()
{
    alpha = 0.0f;
    for (int t = 0; t < 4.0f * ro; t++)
    {
        int v = GetSignalValue(t);
        signalVector.push_back(v);
        OscillateFeedForward(v, t);
        f_vector1.push_back(log(V / U));
    }

    signalVector.clear();
    resultVector.clear();

    alpha = 0.001f;
    for (int t = (int)(4.0f * ro); t < (int)(4.0f * ro + TOTAL_TIME); t++)
    {
        int v = GetSignalValue(t);
        signalVector.push_back(v);
        OscillateFeedForward(v, t);
        getNetwork()->UpdateWeights();

        f_vector1.push_back(log(V / U));

        double correlation = GetCorrelationTrace(signalVector, resultVector, t);
        corVector1.push_back(correlation);
    }

    string ro_string = to_string(ro);
    write_csv("data/" + ro_string + " values.csv", {{"Values", resultVector}});
    write_csv("data/" + ro_string + " F.csv", {{"Values", f_vector1}});
    write_csv("data/" + ro_string + " Cor.csv", {{"Values", corVector1}});
}

void SFA::TrainTwoInvariances()
{
    cout << "started training" << endl;
    alpha = 0.0f;
    for (int t = 0; t < 4.0f * ro; t++)
    {
        pair<int, int> v = GetSignalTuple(t);
        signalVector1.push_back(v.first);
        signalVector2.push_back(v.second);
        OscillateFeedForwardTuple(v.first, v.second, t);
    }

    signalVector1.clear(); signalVector2.clear();
    resultVector1.clear(); resultVector2.clear();

    alpha = 0.001f;
    for (int t = (int)(4.0f * ro); t < (int)(4.0f * ro + TOTAL_TIME); t++)
    {
        pair<int, int> v = GetSignalTuple(t);
        signalVector1.push_back(v.first);
        signalVector2.push_back(v.second);
        OscillateFeedForwardTuple(v.first, v.second, t);
        getNetwork()->UpdateWeights();
    }

    write_csv("data/output1.csv", {{"Values", resultVector1}});
    write_csv("data/output2.csv", {{"Values", resultVector2}});
    write_csv("data/weight_vector.csv", {{"j", getNetwork()->GetWeights()}});
    write_csv("data/signal1.csv", {{"Values", signalVector1}});
    write_csv("data/signal2.csv", {{"Values", signalVector2}});
    write_csv("data/cor1.csv", {{"Values", corVector1}});
    write_csv("data/cor2.csv", {{"Values", corVector2}});
}

int SFA::GetSignalValue(int time)
{
    int j = round(51.0f + 50.0f * sin(M_PI / 180.0f * double(time) * 360.0f / ro));
    // Optimization: avoid clear/push_back
    fill(input_vector.begin(), input_vector.end(), 0.0f);
    if (j >= 1 && j <= (int)NUM_INPUT_NEURONS_X) {
        input_vector[j - 1] = 1.0f;
    }
    return j;
}

pair<int, int> SFA::GetSignalTuple(int time)
{
    float phi = (float)M_PI / 180.0f * time * 17.0f / 360.0f;
    int j1 = round(26.0f + 25.0f * sin((M_PI / 180.0f * time * 360.0f / ro) + phi));
    int j2 = round(26.0f + 25.0f * sin((M_PI / 180.0f * time * 360.0f / ro) - phi));

    fill(input_vector.begin(), input_vector.end(), 0.0f);
    if (j1 >= 1 && j1 <= (int)NUM_INPUT_NEURONS_X && j2 >= 1 && j2 <= (int)NUM_INPUT_NEURONS_Y) {
        input_vector[(j2 - 1) * NUM_INPUT_NEURONS_X + (j1 - 1)] = 1.0f;
    }
    return {j1, j2};
}

double SFA::CalculateDelWeights(int i)
{
    double alphaV = alpha / V;
    double alphaU = alpha / U;

    double dely = y - y_bar;
    double delyt = y - y_tilde;
    double hebbian = alphaV * dely * (input_vector[i] - x_bar_vector[i]);
    double antihebbian = -1.0f * (alphaU * delyt * (input_vector[i] - x_tilde_vector[i]));
    del_weight_vector[i] = hebbian + antihebbian;

    return del_weight_vector[i];
}

double SFA::CalculateDelWeight(double v, double u, double output, double y_b, double y_ti, int i)
{
    double alphaV = alpha / v;
    double alphaU = alpha / u;

    double dely = output - y_b;
    double delyt = output - y_ti;
    double hebbian = alphaV * dely * (input_vector[i] - x_bar_vector[i]);
    double antihebbian = -1.0f * (alphaU * delyt * (input_vector[i] - x_tilde_vector[i]));

    double del_weight = hebbian + antihebbian;
    return del_weight;
}

double SFA::getYBar(double yz, double y_barz)
{
    return (gamma_long * y_barz + (1.0f - gamma_long) * yz);
}

double SFA::getYTilde(double yz, double y_tildez)
{
    return gamma_short * y_tildez + (1.0f - gamma_short) * yz;
}

double SFA::getXBar(double x_barz, double x)
{
    return gamma_long * x_barz + (1.0f - gamma_long) * x;
}

double SFA::getXTilde(double x_tildez, double x)
{
    return gamma_short * x_tildez + (1.0f - gamma_short) * x;
}

double SFA::getV(double Vz, double Y, double Y_BAR)
{
    return gamma_long * Vz + (1.0f - gamma_long) * pow(Y - Y_BAR, 2);
}

double SFA::getU(double Uz, double Y, double Y_TILDE)
{
    return gamma_long * Uz + (1.0f - gamma_long) * pow(Y - Y_TILDE, 2);
}

double SFA::GetOutput(int signal_value)
{
    resultUni.clear();
    GenerateInputs(signal_value); // Updates input_vector
    feedforward(input_vector);
    getNetwork()->getResults(resultUni);
    return resultUni.back();
}

void SFA::UpdateX(int i)
{
    x_tilde_vector[i] = getXTilde(x_tilde_vector[i], input_vector[i]);
    x_bar_vector[i] = getXBar(x_bar_vector[i], input_vector[i]);
}

double SFA::GetY(int sig_val)
{
    y = GetOutput(sig_val);
    resultVector.push_back(y);
    return y;
}

pair<double, double> SFA::GetYTuple(int val1, int val2, int time_step)
{
    double k = 10.0f;
    pair<double, double> y_tuple = GetOutputTuple(val1, val2);
    y_tuple.second = y_tuple.second + k * GetWeightedAntiHebbian(time_step) * y_tuple.first;
    resultVector1.push_back(y_tuple.first);
    resultVector2.push_back(y_tuple.second);
    return y_tuple;
}

double SFA::GetWeightedAntiHebbian(int time_step)
{
    int lambda_correlation = fmin(20 * ro, time_step);
    vector<double> vector_y1;
    vector<double> vector_y2;

    if ((int)resultVector1.size() > lambda_correlation) {
        vector_y1.assign(resultVector1.end() - lambda_correlation, resultVector1.end());
        vector_y2.assign(resultVector2.end() - lambda_correlation, resultVector2.end());
    } else {
        vector_y1 = resultVector1;
        vector_y2 = resultVector2;
    }

    double wah = -1.0f * pearsoncoeff(vector_y1, vector_y2);
    if (isnan(wah)) wah = 0.0f;
    return wah;
}

pair<double, double> SFA::GetOutputTuple(int val1, int val2)
{
    vector<double> results;
    GenerateInputsFromTuple(val1, val2); // Updates input_vector
    feedforward(input_vector);
    getNetwork()->getResults(results);

    return {results[0], results[1]};
}

void SFA::UpdateNeuron(int neuron_index)
{
    getNetwork()->setDeltaWeight(0, neuron_index, 0, del_weight_vector[neuron_index]);
}

void SFA::UpdateNeuronWithDelta(double dw, int neuron_index, int output_neuron_index)
{
    getNetwork()->setDeltaWeight(0, neuron_index, output_neuron_index, dw);
}

void SFA::OscillateFeedForward(int signal_value, int time_step)
{
    int sig_val = GetSignalValue(time_step);
    for (unsigned int neuron_index = 0; neuron_index < NUM_INPUT_NEURONS_X; neuron_index++)
    {
        UpdateX(neuron_index);
    }
    V = getV(V, y, y_bar);
    U = getU(U, y, y_tilde);
    y = GetY(sig_val);
    y_bar = getYBar(y, y_bar);
    y_tilde = getYTilde(y, y_tilde);

    double* del_weights_ptr = del_weight_vector.data();
    double* net_delta_weights_ptr = getNetwork()->m_deltaWeights.data();
    unsigned topology_next = getNetwork()->m_topology[1];

    double* input_ptr = input_vector.data();
    double* x_bar_ptr = x_bar_vector.data();
    double* x_tilde_ptr = x_tilde_vector.data();

    double alphaV = alpha / V;
    double alphaU = alpha / U;
    double dely = y - y_bar;
    double delyt = y - y_tilde;

    #pragma omp target teams distribute parallel for map(to: input_ptr[0:NUM_INPUT_NEURONS_X], x_bar_ptr[0:NUM_INPUT_NEURONS_X], x_tilde_ptr[0:NUM_INPUT_NEURONS_X]) map(tofrom: del_weights_ptr[0:NUM_INPUT_NEURONS_X], net_delta_weights_ptr[0:NUM_INPUT_NEURONS_X*topology_next])
    for (unsigned int neuron_index = 0; neuron_index < NUM_INPUT_NEURONS_X; neuron_index++)
    {
        double hebbian = alphaV * dely * (input_ptr[neuron_index] - x_bar_ptr[neuron_index]);
        double antihebbian = -1.0 * (alphaU * delyt * (input_ptr[neuron_index] - x_tilde_ptr[neuron_index]));
        double dw = hebbian + antihebbian;
        del_weights_ptr[neuron_index] = dw;
        net_delta_weights_ptr[neuron_index * topology_next] = dw;
    }
}

void SFA::OscillateFeedForwardTuple(int signal_value1, int signal_value2, int time_step)
{
    unsigned total_inputs = NUM_INPUT_NEURONS_X * NUM_INPUT_NEURONS_Y;
    #pragma omp simd
    for (unsigned int index = 0; index < total_inputs; index++)
    {
        UpdateX(index);
    }

    V1 = getV(V1, y1, y1_bar);
    V2 = getV(V2, y2, y2_bar);
    U1 = getU(U1, y1, y1_tilde);
    U2 = getU(U2, y2, y2_tilde);

    pair<double, double> y_tuple = GetYTuple(signal_value1, signal_value2, time_step);

    y1 = y_tuple.first;
    y2 = y_tuple.second;

    y1_bar = getYBar(y1, y1_bar);
    y1_tilde = getYTilde(y1, y1_tilde);

    y2_bar = getYBar(y2, y2_bar);
    y2_tilde = getYTilde(y2, y2_tilde);

    double* net_delta_weights_ptr = getNetwork()->m_deltaWeights.data();
    unsigned topology_next = getNetwork()->m_topology[1];

    double* input_ptr = input_vector.data();
    double* x_bar_ptr = x_bar_vector.data();
    double* x_tilde_ptr = x_tilde_vector.data();

    double alphaV1 = alpha / V1;
    double alphaU1 = alpha / U1;
    double dely1 = y1 - y1_bar;
    double delyt1 = y1 - y1_tilde;

    double alphaV2 = alpha / V2;
    double alphaU2 = alpha / U2;
    double dely2 = y2 - y2_bar;
    double delyt2 = y2 - y2_tilde;

    #pragma omp target teams distribute parallel for map(to: input_ptr[0:total_inputs], x_bar_ptr[0:total_inputs], x_tilde_ptr[0:total_inputs]) map(tofrom: net_delta_weights_ptr[0:total_inputs*topology_next])
    for (unsigned int index1 = 0; index1 < total_inputs; index1++)
    {
        double hebbian1 = alphaV1 * dely1 * (input_ptr[index1] - x_bar_ptr[index1]);
        double antihebbian1 = -1.0 * (alphaU1 * delyt1 * (input_ptr[index1] - x_tilde_ptr[index1]));
        net_delta_weights_ptr[index1 * topology_next + 0] = hebbian1 + antihebbian1;

        double hebbian2 = alphaV2 * dely2 * (input_ptr[index1] - x_bar_ptr[index1]);
        double antihebbian2 = -1.0 * (alphaU2 * delyt2 * (input_ptr[index1] - x_tilde_ptr[index1]));
        net_delta_weights_ptr[index1 * topology_next + 1] = hebbian2 + antihebbian2;
    }
}

vector<double> SFA::GenerateInputs(int number_to_encode)
{
    fill(input_vector.begin(), input_vector.end(), 0.0f);
    if (number_to_encode >= 1 && number_to_encode <= (int)NUM_INPUT_NEURONS_X) {
        input_vector[number_to_encode - 1] = 1.0f;
    }
    return input_vector;
}

vector<double> SFA::GenerateInputsFromTuple(int number1, int number2)
{
    fill(input_vector.begin(), input_vector.end(), 0.0f);
    if (number1 >= 1 && number1 <= (int)NUM_INPUT_NEURONS_X && number2 >= 1 && number2 <= (int)NUM_INPUT_NEURONS_Y) {
        input_vector[(number2 - 1) * NUM_INPUT_NEURONS_X + (number1 - 1)] = 1.0f;
    }
    return input_vector;
}

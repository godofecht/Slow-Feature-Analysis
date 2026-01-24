#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include "NN.h"
#include "Network.h"
#include "sfa.h"
#include "stats.h"

void test_stats() {
    std::cout << "Testing Stats..." << std::endl;
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 4, 6, 8, 10};
    double r = pearsoncoeff(x, y);
    assert(std::abs(r - 1.0) < 1e-6);

    std::vector<double> z = {5, 4, 3, 2, 1};
    r = pearsoncoeff(x, z);
    assert(std::abs(r + 1.0) < 1e-6);
    std::cout << "Stats tests passed!" << std::endl;
}

void test_neuron() {
    std::cout << "Testing Neuron..." << std::endl;
    Neuron n(2, 0);
    assert(n.getIndex() == 0);
    n.setOutputVal(0.5);
    assert(n.getOutputVal() == 0.5);
    std::cout << "Neuron tests passed!" << std::endl;
}

void test_network() {
    std::cout << "Testing Network..." << std::endl;
    std::vector<unsigned> topology = {2, 3, 1};
    Network net(topology);
    assert(net.m_layers.size() == 3);
    assert(net.m_layers[0].size() == 2);
    assert(net.m_layers[1].size() == 3);
    assert(net.m_layers[2].size() == 1);

    std::vector<double> inputs = {0.5, 0.8};
    net.feedForward(inputs);
    std::vector<double> results;
    net.getResults(results);
    assert(results.size() == 1);
    std::cout << "Network tests passed!" << std::endl;
}

void test_sfa() {
    std::cout << "Testing SFA..." << std::endl;
    SFA sfa(50.0, 1);
    // Simple test to see if it doesn't crash
    int sig = sfa.GetSignalValue(0);
    assert(sig >= 0);
    double y = sfa.GetY(sig);
    (void)y;
    std::cout << "SFA tests passed!" << std::endl;
}

int main() {
    test_stats();
    test_neuron();
    test_network();
    test_sfa();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}

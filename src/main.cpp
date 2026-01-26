#include <iostream>
#include "sfa.h"

std::vector<double> ro_VALS = {50.0f, 100.0f, 200.0f, 450.0f, 900.0f, 1800.0f, 3600.0f};

int main() {
    for (double ro : ro_VALS) {
        std::cout << "Training for ro = " << ro << "..." << std::endl;
        SFA sfa(ro, 1);
        sfa.Train();
    }

    // SFA sfa_two(450.0f, 2);
    // sfa_two.TrainTwoInvariances();

    return 0;
}

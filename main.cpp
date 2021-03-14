
// Your First C++ Program



#include <iostream>
#include "sfa.h"

std::vector<double> ro_VALS = {50.0f,100.0f,200.0f,450.0f,900.0f,1800.0f,3600.0f};


int main() {

  SFA sfa1((ro_VALS[0]),1);
  sfa1.Train();

  SFA sfa2((ro_VALS[1]),1);
  sfa2.Train();

  SFA sfa3((ro_VALS[2]),1);
  sfa3.Train();

  SFA sfa4((ro_VALS[3]),1);
  sfa4.Train();

  SFA sfa5((ro_VALS[4]),1);
  sfa5.Train();

  SFA sfa6((ro_VALS[5]),1);
  sfa6.Train();

  SFA sfa7((ro_VALS[6]),1);
  sfa7.Train();


//   sfa.TrainTwoInvariances();

    return 0;
}


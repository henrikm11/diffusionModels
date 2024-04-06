//predictor.h


#pragma once

#include<vector>
#include "is_predictor.h"


namespace predictor{

class Model{
public:
    Model(int val=42);
    double predict(double) const;
    double predict(double, double) const;
    std::vector<double> predict(const std::vector<double>&, double) const;

private:
    int val_;
};

};

/*
template<>
struct is_predictor<predictor::Model>{
    const static bool value = true;
};
*/




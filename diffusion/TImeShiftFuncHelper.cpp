//TimeShiftFuncHelper.cpp
//implementation of diffusion::TimeShiftFuncHelper.cpp

#include "FuncHelper.h"

namespace diffusion{

TimeShiftFuncHelper::TimeShiftFuncHelper(
    const FuncHelper& original,
    double shift,
    double speed
)
    :originalFct_(original.clone()),
    shift_(shift),
    speed_(speed)
{}

TimeShiftFuncHelper::TimeShiftFuncHelper(
    const std::unique_ptr<FuncHelper>& original,
    double shift,
    double speed
)
    :originalFct_(original->clone()),
    shift_(shift),
    speed_(speed)
{}

TimeShiftFuncHelper::TimeShiftFuncHelper(
    const TimeShiftFuncHelper& other,
    double factor,
    double power
)
    :FuncHelper(
        other.multiply_,
        other.integral_,
        other.factor_*factor,
        other.power_*power
    ),
    originalFct_(other.originalFct_->clone()),
    shift_(other.shift_),
    speed_(other.speed_)
{}

TimeShiftFuncHelper::TimeShiftFuncHelper(
    const TimeShiftFuncHelper& other,
    bool multiply,
    bool integral,
    double factor,
    double power
)
    :FuncHelper(
        multiply,
        integral,
        other.factor_*factor,
        other.power_*power
    ),
    originalFct_(other.originalFct_->clone()),
    shift_(other.shift_),
    speed_(other.speed_)
{}

std::unique_ptr<FuncHelper> TimeShiftFuncHelper::clone(void) const {
    return std::unique_ptr<FuncHelper>(new TimeShiftFuncHelper(*this));
}

std::unique_ptr<FuncHelper> TimeShiftFuncHelper::modifiedClone(
    double factor,
    double power
) const {
    return std::unique_ptr<FuncHelper>(new TimeShiftFuncHelper(*this, factor, power));

}

std::unique_ptr<FuncHelper> TimeShiftFuncHelper::modifiedClone(
    bool multiply, 
    bool integral, 
    double factor, 
    double power
) const {
    return std::unique_ptr<FuncHelper>(new TimeShiftFuncHelper(*this, multiply, integral, factor, power));
}


	
double TimeShiftFuncHelper::operator()(double t) const {
    return originalFct_->operator()(shift_+speed_*t);
}

double TimeShiftFuncHelper::operator()(double x, double t) const {
    return originalFct_->operator()(x,shift_+speed_*t);
}
    
std::vector<double> TimeShiftFuncHelper::operator()(
    const std::vector<double>& X, 
    double t
) const {
    return originalFct_->operator()(X,shift_+speed_*t);

}
   
}
#include <iostream>
using namespace std;

template <typename numberType>
struct isMultiple
{
    numberType divisor;
    
    isMultiple (const numberType & pDivisor) {
        divisor = pDivisor;
    }

    bool operator () (const numberType& element ) const
    {
        return (element % divisor ) == 0;
    }
};

int main()
{
    int lDivisor = 3;
    int lDivident = 21123131;

    isMultiple<int> divident(lDivisor);

    if (divident(lDivident)) {
        cout << lDivident << " is a multiple of " << lDivisor;
    } else {
        cout << lDivident << " is NOT a multiple of " << lDivisor;
    }
    return 0;
}

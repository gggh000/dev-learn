#include <iostream>
using namespace std;

// template with default params; int & double;
template <typename T1, typename T2>
class holdsPair
{
private:
    T1 value1;
    T2 value2;
public:

    // Constructor that initializes member variables.

    holdsPair (const T1 & val1, const T2 & val2)
    {
        value1 = val1;
        value2 = val2;
    }

    const T1 & getFirstValue() const 
    { return value1; }

    const T2 & getSecondValue() const
    { return value2; }
};

int main()
{
    holdsPair<int, double> pairIntDbl ( 6, 1.99 ); 
    holdsPair<short, const char *>pairShortStr (25, "Learn template, love C++");

    cout << "The first object contains..." << endl;
    cout << "Value1: " << pairIntDbl.getFirstValue() << endl;
    cout << "Value2: " << pairIntDbl.getSecondValue() << endl;

    cout << "The second object contains..." << endl;
    cout << "Value1: " << pairShortStr.getFirstValue() << endl;
    cout << "Value2: " << pairShortStr.getSecondValue() << endl;
}

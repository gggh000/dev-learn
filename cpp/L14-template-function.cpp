#include <iostream>
#include <string>
using namespace std;

template <typename Type>
const Type& getMax(const Type& value1, const Type& value2)
{
    if (value1 > value2)
        return value1;
    else
        return value2;
}

template <typename Type>
void displayComparison(const Type& value1, const Type& value2)
{
    cout << "getmax(" << value1 << ", " << value2 << ") = ";
    cout << getMax(value1, value2) << endl;
}

int main()
{
    int num1 = 12, num2 = 113;
    displayComparison<int>(num1, num2);

    double d1 = 123.4, d2 = 13.2;
    displayComparison<double>(d1, d2);
    return 0;
}

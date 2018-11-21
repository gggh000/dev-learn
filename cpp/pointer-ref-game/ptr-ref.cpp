#include <iostream>
using namespace std;

int f1_copy(int pInt) 
{
    return 0;    
}

int main()
{
    int int1 = 100;
    int int2 = int1;
    int * intPtr1 = &int1;
    int & intRef = int1;

    cout << "address of int1        :" << &int1 << endl;
    cout << "address of int2        :" << &int2 << endl;
    cout << "address held by intPtr1:" << intPtr1 << endl;
    cout << "address of intRef      :" << &intRef << endl;
}

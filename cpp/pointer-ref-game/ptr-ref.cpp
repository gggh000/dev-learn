#include <iostream>
using namespace std;

int f1_ref(int & pInt)
{
    cout << "f1_ref: &pInt/pInt:   :" << &pInt << ", " << pInt << endl;
    return 0;    
}

int f1_ptr(int * pInt)
{
    cout << "f1_ref: pInt/*pInt:   :" << pInt << ", " << *pInt << endl;
    return 0;    
}

int f1_copy(int pInt) 
{
    cout << "f1_copy: &pInt/pInt:   :" << &pInt << ", " << pInt << endl;
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
    
    f1_copy(int1);
    f1_copy(int2);
    f1_ref(int1);
    f1_ref(int2);
    f1_ptr(&int1);
    f1_ptr(&int2);
}

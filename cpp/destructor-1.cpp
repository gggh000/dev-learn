/*
This example illustrates simple constructor, desctructor and copy constructor.
For copy constructor, deep copy is performed on buffer int pointer called num when it is copied.
The string version of this example, implemented in destructor.cpp is crashing upon release
of buffer for original one.
*/

#include <iostream>
#include <string.h>
using namespace std;

class myString
{
private:

public:
    int * num;
    myString(int pNum)
    {
        cout << "Constructor called." << endl;

        if(pNum != NULL) {
            num = new int(50);
            cout << "num points to: 0x" << hex;
            cout << (unsigned int*) num << endl;
        } else {
            num = new int(100);
            cout << "num points to: 0x" << hex;
            cout << (unsigned int*) num << endl;
        }
    }    

    myString(myString& copySource)
    {
        num = NULL;
        cout << "Copy constructor: copying from copySource" << endl;

        if (copySource.num != NULL)
        {
            num = new int(10);
            cout << "num points to: 0x" << hex;
            cout << (unsigned int*) num << endl;
        }
    }
    
    ~myString()
    {
        cout << "Destructor called, deleting num ptr: " << num << ": 0x" << hex <<  (unsigned int*)num << endl;
        delete num;
    }
};

void useMyString(myString str)
{
    cout << "entered useMyString()" << endl;
}

int main()
{
    myString str1(24);
    // useMyString(str1);    
    myString str2(str1);

    cout << "str1: " << " " << (str1.num) << endl;
    cout << "str2: " << " " << (str2.num) << endl;
}

#include <iostream>
using namespace std;

class sPtr
{
    int * data;

public:

    // constructor. 

    sPtr (int pData = NULL) {
        cout << "sPtr constructor." << endl;
        data = new int(pData);
    }

    // destructor.

    ~sPtr () { 
        cout << "sPtr desctructor." << endl;
        delete data; 
    }

    // accessory.
        
    int getData() { return *data; } 

    // print info

    void printInfo()
    {
        cout << "object addr:   " << hex << this << endl;
        cout << "data ptr addr: " << hex << data << endl;

        if (data != NULL) {
            cout << "data value:            " << dec << *data << endl; 
        } else { 
            cout << "data value is NULL.    " << endl;
        }
    }

private:
    
};

int main()
{
    sPtr * int1;    
    int1 = new sPtr(100);

    cout << "int1: " << int1->getData() << endl;
    int1->printInfo();
    delete int1;
    return 0;
}

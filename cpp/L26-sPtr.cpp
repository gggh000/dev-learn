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

private:
    
};

int main()
{
    sPtr * int1;    
    int1 = new sPtr(100);

    cout << "int1: " << int1->getData() << endl;
    delete int1;
    return 0;
}

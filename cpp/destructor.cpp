#include <iostream>
#include <string.h>
using namespace std;

class myString
{
private:
    char * buffer;
    int * num;
public:
    myString(const char * initString)
    {
        cout << "Constructor called." << endl;

        if(initString != NULL) 
        {
            buffer = new char [strlen(initString) + 1];
            strcpy(buffer, initString);
            num = new int(10);
        }
        else
            buffer = NULL;
    }    

    myString(const myString& copySource)
    {
        buffer = NULL;
        cout << "Copy constructor: copying from copySource" << endl;

        if (copySource.buffer != NULL)
        {
            buffer = new char [strlen(copySource.buffer) + 1];
            strcpy(buffer, copySource.buffer);

            cout << "buffer points to: 0x" << hex;
            cout << (unsigned int*) buffer << endl;
        }
    }
    
    ~myString()
    {
        cout << "Destructor called, deleting buffer: " << buffer << endl;
        delete [] buffer;
        delete num;
    }

    int getLength() {
        return strlen(buffer);
    }

    string getString() {
        return buffer;
    }
};

void useMyString(myString str)
{
    cout << "entered useMyString()" << endl;
    cout << "String buffer in myString is " << str.getLength();
    cout << " characters long" << endl;

    cout << "buffer contains: " << str.getString() << endl;
}

int main()
{
    myString str1("str1");
    useMyString(str1);    
}

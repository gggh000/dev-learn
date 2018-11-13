#include <iostream>
#include <string.h>
using namespace std;

class myString
{
private:
    char * buffer;
public:
    int * num;
    myString(char * initString)
    {
        cout << "Constructor called." << endl;

        if(initString != NULL) 
        {
            buffer = new char [strlen(initString) + 1];
            strcpy(buffer, initString);
            num = new int(10);
            cout << "buffer points to: 0x" << hex;
            cout << (unsigned int*) buffer << endl;
            cout << "num points to: 0x" << hex;
            cout << (unsigned int*) num << endl;
        }
        else
            buffer = NULL;
    }    

    myString(myString& copySource)
    {
        buffer = NULL;
        num = NULL;
        cout << "Copy constructor: copying from copySource" << endl;

        if (copySource.buffer != NULL)
        {
            buffer = new char [strlen(copySource.buffer) + 20];
            strcpy(buffer, copySource.buffer);
            buffer = "str1 copied";

            cout << "buffer points to: 0x" << hex;
            cout << (unsigned int*) buffer << endl;
            
        }
        if (copySource.num != NULL)
        {
            num = new int(10);
            cout << "num points to: 0x" << hex;
            cout << (unsigned int*) num << endl;
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
    // useMyString(str1);    
    myString str2(str1);

    cout << "str1: " << str1.getString() << " " << *(str1.num) << endl;
    cout << "str2: " << str2.getString() << " " << *(str2.num) << endl;
}

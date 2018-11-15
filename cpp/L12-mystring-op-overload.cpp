#include <iostream>
using namespace std;
#include <string.h>

class MyString
{
private:
    char * buffer;

public:
    MyString(const char * initialInput)
    {
        if (initialInput != NULL) 
        {
            buffer = new char [strlen(initialInput) + 1];
            strcpy(buffer, initialInput);
        }
        else
            buffer = NULL;
    }

    // copy assignment constructor.

    MyString & operator=(const MyString & copySource)
    {
        if ((this != &copySource) && copySource.buffer != NULL)
        {
            if (buffer != NULL)
                delete[] buffer;

            buffer = new char [strlen(copySource.buffer) + 1];
            strcpy(buffer, copySource.buffer);
        }
        return *this;
    }   

    int GetLength() const
    {
        return strlen(buffer);
    }

    // copy constructor.

    operator const char*()
    {
        return buffer;
    }

    // constructor

    ~MyString()
    {
        delete[] buffer;
    }    

    const char & operator[] (int index) const
    {
        if (index < GetLength())        
            return buffer[index];        
    }

};

int main()
{
    MyString string1("Hello");
    MyString string2(" World");

    cout << "Before assignment: " << endl;
    cout << string1 << string2 << endl;

    string2 = string1;

    cout << "After assignment: " << endl;
    cout << string1 << string2 << endl;

    // Subscript operator.

    cout << "Type a statement: ";
    string strInput;
    getline(cin, strInput);
    
    MyString youSaid(strInput.c_str());
    
    cout << "Using operator[] for displaying your input: " << endl;
    
    for (int index = 0; index < youSaid.GetLength(); ++index)
    {
        cout << youSaid[index] << " ";
    }    
    
    cout << endl;
}



    

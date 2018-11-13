#include <iostream>
#include <string>
using namespace std;

class Human 
{
private:
    string name;
    int age;

public:
    //Human();
    Human(string name, int age);
    void introduceSelf();
};

/*
Human::Human() {
    //name = "Default name-1";
    //age = 101;
    cout << "Default constructor set the name and age to " << name << ", " << age << endl;
}
*/

Human::Human(string pName = "default name", int pAge = 100)
{
    cout << "Overloaded constructor setting name and age by parameter to " << pName << ", " << pAge << endl;
    age = pAge;
    name = pName;
}

void Human::introduceSelf() {
    cout << "I am " + name << " and am ";
    cout << age << " years old" << endl;
}

int main() 
{
    Human firstMan;
    firstMan.introduceSelf();
    Human secondMan;
    secondMan.introduceSelf();
}

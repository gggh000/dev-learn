#include <iostream>
#include <string>
using namespace std;

class Human 
{
private:
    string name;
    int age;

public:
    Human();
    Human(string name, int age);
    void introduceSelf();
};

Human::Human() {
    name = "Default name";
    age = 100;
    cout << "Default constructor set the name and age to " << name << ", " << age << endl;
}

Human::Human(string pName, int pAge)
{
    cout << "Overloaded constructor setting name and age by parameter to " << name << ", " << age << endl;
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
}

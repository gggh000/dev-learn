/*
Simple example of public inheritance.
The Fish method is also overridden in derived class.
main calls both Fish overriden method and explicitly calling the base class Fish() using scope
resolution operator ::.
*/
#include <iostream>
using namespace std;
static bool DEBUG = 0;

class Fish 
{
private:

protected:
    bool isFreshWaterFish;

public:
    Fish(bool isFreshWater)
    {
        if (DEBUG == true) 
        {
            cout << "Fish::Fish constructor called." << endl;
        }
        isFreshWaterFish = isFreshWater;
    }

    void Swim()
    {
        if (DEBUG == true) 
        {
            cout << "Fish::Swim method called." << endl;
        }
        if (isFreshWaterFish) 
            cout << "Can swim in lake." << endl;
        else
            cout << "Can swim in sea." << endl;
    }
};

class Tuna : public Fish
{
public:
    Tuna(): Fish(false) 
    {
        if (DEBUG == true) 
        {
            cout << "Tuna::Tuna constructor called." << endl;
        }
    }

    void Swim()
    {
        cout << "Tuna swims very fast." << endl;
    }    
};

class Carp : public Fish
{
public:
    Carp(): Fish(true) 
    {
        if (DEBUG == true) 
        {
            cout << "Carp::Carp constructor called." << endl;
        }
    }

    void Swim()
    {
        cout << "Tuna swims real slow." << endl;
    }    
};

int main()
{
    Carp myLunch;
    Tuna myDinner;
    cout << "About my food" << endl;

    cout << "Lunch: ";
    myLunch.Swim();
    myLunch.Fish::Swim();
    cout << "Dinner: ";
    myDinner.Swim();
    myDinner.Fish::Swim();
}

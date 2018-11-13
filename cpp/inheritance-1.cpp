/*Simple example of private inheritance */

#include <iostream>
using namespace std;

class Motor
{
public:
    void switchIgnition()
    { cout << "Ignition on." << endl; }

    void pumpFuel()
    { cout << "Fuel in cylinders." << endl; }

    void fireCylinders()
    { cout << "Engine started." << endl; }
};

class Car : private Motor 
{
public:
    void move()
    {
        switchIgnition();
        pumpFuel();
        fireCylinders();
    }
};

int main()
{
    Car myCar;
    myCar.move();

    return 0;
}

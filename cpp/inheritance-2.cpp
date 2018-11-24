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

class Car : protected Motor 
{
public:
    void move()
    {
        switchIgnition();
        pumpFuel();
        fireCylinders();
    }
};

class RaceCar : protected Car
{
public:
    void move()
    {
        switchIgnition();
        pumpFuel();
        fireCylinders();
        fireCylinders();
        fireCylinders();
        fireCylinders();
    }
};

int main()
{
    RaceCar myCar;
    myCar.move();
    return 0;
}

#include <iostream>
using namespace std;

class Date
{
private:
    int day, month, year;

public:
    Date (int inMonth, int inDay, int inYear)
        : month(inMonth), day(inDay), year(inYear) {};

    // prefix increment.

    Date & operator ++ ()
    {
        ++day;
        return *this;
    }

    // prefix decrement

    Date & operator -- ()
    {
        --day;
        return *this;
    }

    void displayDate()
    { 
        cout << month << " / " << day << " / " << year << endl; 
    }
};

int main()
{
    Date holiday(12, 25, 2016);
    cout << "The date object is initialized to: ";
    holiday.displayDate();
    
    ++holiday;
    cout << "Date after prefix-increment is: ";
    holiday.displayDate();

    --holiday;
    cout << "Date after a prefix-decrement is: ";
    holiday.displayDate();

    return 0;
}




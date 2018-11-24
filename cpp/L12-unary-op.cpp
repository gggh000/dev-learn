/*
Operator overloading examples for unary operator.
1. ++ prefix
2. postfix ++ 
3. operator const char*() implements the << operator. ostringstream formattedDate is used
by feeding the integers. And resulting string in formattedDate is assigned to private variable
dateInString because formattedDate is local to a function: operator const char*() and will
be destroyed when function returns.
*/

#include <iostream>
#include <sstream>
#include <string>

using namespace std;

class Date
{
private:
    int day, month, year;
    string dateInString;

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

    operator const char*()
    {
        ostringstream formattedDate;
        formattedDate << month << " / " << day << " / " << year;

        dateInString = formattedDate.str();
        return dateInString.c_str();
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

    cout << "Print using cout operator << :";
    cout << holiday << endl;

    string strHoliday (holiday);
    strHoliday = Date(11, 11, 2016);

    cout << "Printing holiday: ";
    cout << strHoliday << endl;

    // Implementation of smart pointer. 
    
    /*
    unique_ptr<int> smartIntPtr(new int);
    *smartIntPtr = 42;

    cout << "Integer value pointed by smart pointer is: " << *smartIntPtr << endl;

    unique_ptr<Date> smartHoliday (new Date(12, 25, 2016));
    cout << "The new instance of date contains: ";

    smartHoliday->displayDate();
    */
    
    return 0;
}




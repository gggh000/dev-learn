#include <list>
#include <mutex>
#include <algorithm>
#include <thread> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>

using namespace std;

std::mutex mutex_sum;
int sum = 0;

//  This function is called per thread and therefore duplicated for each threads
//  that are spawned.  

void hello(int pId, int pStat[])
{
    // Sleep random period.

    int pNum = rand() % 4 + 2;
    int sleep_duration = 1000000 * (rand() % 2 + 0);
    sleep_duration = 0;
    std::cout << pId << ": Hello CONCURRENT WORLD, sleeping for " << sleep_duration << endl;
    usleep(sleep_duration);

    // Create lock guard.
    //std::lock_guard<std::mutex> sum_guard(mutex_sum);
    // myfile is used to  output the sum. It is named as file-<PID>.
    // Sum is incremented by each thread's random number and gets published.
    // Here is the idea of race condition being created deliberedly;
    // - each line in the output file reflected 
    //      1. sum before adding  pNum.
    //      2. pNum being added to sum.
    //      3. sum after adding pNum.
    //  iF race condition occurs, it is likely due  to the after another thread has updated
    // the sum causing sum (before add) + pNum != sum (after add).
    // We run it enought number threads to increase the likelihood  of this happening.
    // - We can proof of concept by now protecting the summation operation with mutex and 
    // Verify it does not happen. 

    ofstream myfile;
    ostringstream oss;
    oss << "file-" << pId;
    myfile.open(oss.str());
    myfile << "thread: " << pId << ": R0: " << sum;
    sum += pNum;
    myfile << " + " << pNum << " = " << sum << endl;
    myfile.close();
    pStat[pId]  = 1;
    std::cout << pId << ": Done sleeping exiting now..." << endl;
}

class data_wrapper
{
private:
    int sum;      
    std::mutex m;

public:
    data_wrapper()
    {
        cout << "data_wrapper constructor." << endl;
        sum = 0;
    }

    void add_using_mutex(int pThreadId, int pNum)
    {
        std::lock_guard<std::mutex> guard(m);

        // read current value.
        // add random value.
        // read back sum.
        // (if no race condition, R + ADD + R should be consistent.

        cout << "thread: " << pThreadId << ": R0: " << sum;
        sum += pNum;
        cout << " + " << pNum << " = " << sum << endl;
    }
}; 

int main()
{
    // Declare, initialize variables.

    int i;
    const int CONFIG_THREAD_COUNT = 2;
    int stat[CONFIG_THREAD_COUNT];
    int sum = 0;

    // launch threads the CONFIG_THREAD_COUNT instance, with hello function, i index and stat array. 

    for ( i = 0; i < CONFIG_THREAD_COUNT; i ++ ) {
        stat[i] = 0;
        std::thread t(hello, i, stat);
        t.detach();
    }

    cout << "Checking thread status-s..." << endl;

    // Check thread statuses,  if sum is same as CONFIG_THREAD_COUNT, all thread has completed
    // their job.

     while (sum != CONFIG_THREAD_COUNT)  {
        sum = 0;

        for (i = 0; i < CONFIG_THREAD_COUNT; i++) {
            cout << stat[i] << ", ";
            sum += stat[i];
        }

        cout << "main(): sum: " << sum << ". waiting for all threads to finish..." << endl;
        usleep(5 * 1000000);
    }

    return 0;
}

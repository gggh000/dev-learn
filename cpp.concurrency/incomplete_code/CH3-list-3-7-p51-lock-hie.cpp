/*
Lock hierarch example from C++ concurrency in action by Anthony Williams
Chapter 3
Listing 3.7
Using a lock hierarchy to prevent deadlock.

This is untested and non-functional code. 
Improvement to functional code can be made later.

*/

//  Create two levels of mutex.

hierarchical_mutex high_level_mutex (10000);
hierarchical_mutex low_level_mutex (5000);

//  Function for doing lower level lock activity.

int do_low_level_stuff();

//  Acquires lock on low level mutext and perform lower level job. 

int low_level_func() {
    std::lock_guard<hierarchical_mutex> lk(low_level_mutex);
    return do_level_stuff();
}

//  Function for performing higher level stuff.

void high_level_stuff(int some_param);

//  Acquire lock on mutext for performaing higher level activity.

void high_level_func() {
    std::lock_guard<hierarchical_mutex> lk (high_level_mutex);

    // With lock acquired on high_level_mutex, the call to low_level_func passed 
    // as a paramteter to high_level_stuff will set the lock on low_level_mutext @ 5000.
    // This means hgih_level_func(0), this function sets high level mutex and then low level mutex
    // in order. 

    high_level_stuff(low_level_func());
}

//  thread A will do a high level activity.

void thread_a() {
    high_level_func();
}

hierarchical_mutex other_mutex (100);
void do_other_stuff();

void other_stuff() {
    high_level_func();
    do_other_stuff();
}

// thread B will do a lower level activity.

void thread_b() {
    std::lock_guard<hierarchical_mutex> lk(other_mutex);
    other_stuff();
}

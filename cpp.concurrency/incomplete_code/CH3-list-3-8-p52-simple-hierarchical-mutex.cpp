/*
Hierarchical mutex example from C++ concurrency in action by Anthony Williams
Chapter 3
Listing 3.8
This implements the hierarchical mutex not in the standard library but easy to implement one.
It can be used with the lock_guard (RAII) since it implements lock(), unlock() and try_lock().

This is untested and non-functional code.
Improvement to functional code can be made later.

*/

class hierarchical_mutex {
    std::mutex internal_mutex;
    unsigned long const hierarchy value;
    unsigned long previous_hierarchy_value;

    // thread_local keyword specifies storage duration.

    static thread_local unsigned long this_thread_hierarchy_value;

    /* Checks for hierarchy violation. If the new hierarchy_value that is initialized by constructor
    is greater than this thread's, then it throws exception. Otherwise, it is allowed to continue. */
   
    void check_for_hierarchy_violation() {
        if (this_thread_hierarchy_value  <= hierarchy_value) {
            throw std::logic_error("Mutex hierarchy violated.");
        }
    }

    /*  Updates previous hierarchy value to current this_thread_hierarchy_value for locking. 
    Restore is done during unlocking. */

    void update_hierarchy_value() {
        previous_hierarchy_value = this_thread_hierarchy_value;
        this_thread_hierarchy_value=hierarchy_value;
    }

public:

    /* New hierarchical value is introduced by caller later to be checked against the 
    this thread's hierarchical value for violation. This is done during lock. */

    explicit hierarchial_mutex(unsigned long value):
        hierarchy_value(value),
        previous_hierarchy_value(0)
    {}


    /* Locking mechanism, verify first by checking newly initialized hierarchy_value is smaller 
    than this_thread_hierarchy_value to be able to continue, otherwise exception is thrown through
    check_for_hierarchy_violation. 
    */

    void lock() {
        check_for_hierarchy_violation();
        internal_mutex.lock();
        update_hierarchy_value();
    }

    /*  Unlocking mechanism for mutex. Restore the previous hierarchy value and unlocks. */

    void unlock() {
        this_thread_hierarchy_value = previous_hierarchy_value;
        internal_mutex.unlock();

    }

    bool try_lock() {
        check_for_hierarchy_violation();

        if (!internal_mutex.try_lock())
            return false;

        update_hierarchy_value();
        return true;    
    }
};

// this_thread_hierarchy_value is set to max value so that can mutex can be locked initially. 
// Because it is declared as thread_local, every thread has its own copy. 

thread_local unsigned_long hierarchical_mutex::this_thread_hierarchy_value(ULONG_MAX);

    


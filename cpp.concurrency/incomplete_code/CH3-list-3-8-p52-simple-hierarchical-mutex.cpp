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
    static thread_local unsigned long this_thread_hierarchy_value;

    void check_for_hierarchy_violation() {
        if (this_thread_hierarchy_value  <= hierarchy_value) {
            throw std::logic_error("Mutex hierarchy violated.");
        }
    }

    void update_hierarchy_value() {
        previous_hierarchy_value = this_thread_hierarchy_value;
        this_thread_hierarchy_value=hierarchy_value;
    }

public:
    explicit hierarchial_mutex(unsigned long value):
        hierarchy_value(value);
        previous_hierarchy_value(0)
    {}

    void lock() {
        check_for_hierarchy_violation();
        internal_mutex.lock();
        update_hierarchy_value();
    }

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
}
    

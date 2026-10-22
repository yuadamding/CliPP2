#include "executor.h"
namespace clipp2kernel {
Executor::Executor(int workers) {
    threads.reserve(workers-1);
    try { for(int i=1;i<workers;++i)threads.emplace_back([this]{worker();}); }
    catch(...) {
        { std::lock_guard<std::mutex> lock(mutex);stopping=true; }
        ready.notify_all();for(auto& thread:threads)thread.join();throw;
    }
}
Executor::~Executor() {
    {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
    ready.notify_all();for(auto& thread:threads)thread.join();
}
void Executor::consume() {
    try {
        if(each_worker){operation(context,0);return;}
        for(;;){const int i=next.fetch_add(1);if(i>=count)break;operation(context,i);}
    }
    catch(...) {std::lock_guard<std::mutex> lock(mutex);if(!failure)failure=std::current_exception();next=count;}
}
void Executor::worker() {
    unsigned long observed=0;
    for(;;) {
        std::unique_lock<std::mutex> lock(mutex);
        ready.wait(lock,[&]{return stopping||epoch!=observed;});
        if(stopping)return;
        observed=epoch;lock.unlock();consume();lock.lock();
        if(--pending==0)finished.notify_one();
    }
}
}

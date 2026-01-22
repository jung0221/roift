#ifndef GFT_DEBUG_H
#define GFT_DEBUG_H

#include <iostream>
#include <chrono>
#include <iomanip>
#include <map>
#include <string>
#include <cstring>

namespace gft
{

    /**
     * @brief Simple debug timing utility for profiling code sections
     *
     * Usage:
     *   DebugTiming::start("section_name");
     *   // ... code to profile ...
     *   DebugTiming::end("section_name");
     */
    class DebugTiming
    {
    public:
        struct TimingEvent
        {
            std::string name;
            std::chrono::high_resolution_clock::time_point start_time;
            double elapsed_ms;
            int call_count;

            TimingEvent() : elapsed_ms(0.0), call_count(0) {}
        };

        static DebugTiming &getInstance()
        {
            static DebugTiming instance;
            return instance;
        }

        static void start(const std::string &name)
        {
            getInstance()._start(name);
        }

        static void end(const std::string &name)
        {
            getInstance()._end(name);
        }

        static void reset()
        {
            getInstance()._reset();
        }

        static void printSummary()
        {
            getInstance()._printSummary();
        }

        static void setVerbose(bool verbose)
        {
            getInstance().verbose = verbose;
        }

    private:
        std::map<std::string, TimingEvent> events;
        bool verbose;

        DebugTiming() : verbose(
#ifdef _DEBUG
                            true // Enable verbose output only in Debug builds
#else
                            false // Disable verbose output in Release builds
#endif
                        )
        {
        }

        void _start(const std::string &name)
        {
            auto now = std::chrono::high_resolution_clock::now();
            if (events.find(name) == events.end())
            {
                events[name] = TimingEvent();
            }
            events[name].name = name;
            events[name].start_time = now;

            if (verbose)
            {
                std::cout << "[DEBUG] >>> START: " << name << std::endl;
                std::cout.flush();
            }
        }

        void _end(const std::string &name)
        {
            auto now = std::chrono::high_resolution_clock::now();
            if (events.find(name) == events.end())
            {
                return; // Event not started
            }

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - events[name].start_time);
            events[name].elapsed_ms += duration.count();
            events[name].call_count++;

            if (verbose)
            {
                std::cout << "[DEBUG] <<< END: " << name
                          << " | Elapsed: " << std::fixed << std::setprecision(2)
                          << duration.count() << " ms" << std::endl;
                std::cout.flush();
            }
        }

        void _reset()
        {
            events.clear();
        }

        void _printSummary()
        {
            if (events.empty())
            {
                return;
            }

            std::cout << "\n"
                      << std::string(80, '=') << std::endl;
            std::cout << "TIMING SUMMARY - GFT DEBUG" << std::endl;
            std::cout << std::string(80, '=') << std::endl;
            std::cout << std::left << std::setw(40) << "Event Name"
                      << std::right << std::setw(15) << "Total (ms)"
                      << std::setw(15) << "Calls"
                      << std::setw(10) << "Avg (ms)" << std::endl;
            std::cout << std::string(80, '-') << std::endl;

            double total_time = 0.0;
            for (const auto &pair : events)
            {
                const TimingEvent &evt = pair.second;
                if (evt.call_count > 0)
                {
                    double avg = evt.elapsed_ms / evt.call_count;
                    std::cout << std::left << std::setw(40) << evt.name
                              << std::right << std::setw(15) << std::fixed << std::setprecision(2) << evt.elapsed_ms
                              << std::setw(15) << evt.call_count
                              << std::setw(10) << std::fixed << std::setprecision(2) << avg << std::endl;
                    total_time += evt.elapsed_ms;
                }
            }

            std::cout << std::string(80, '-') << std::endl;
            std::cout << std::left << std::setw(40) << "TOTAL TIME"
                      << std::right << std::setw(15) << std::fixed << std::setprecision(2) << total_time
                      << " ms" << std::endl;
            std::cout << std::string(80, '=') << std::endl;
        }
    };

} // namespace gft

// Convenience macros
#define DEBUG_START(name) gft::DebugTiming::start(name)
#define DEBUG_END(name) gft::DebugTiming::end(name)
#define DEBUG_SUMMARY() gft::DebugTiming::printSummary()
#define DEBUG_RESET() gft::DebugTiming::reset()
#define DEBUG_VERBOSE(v) gft::DebugTiming::setVerbose(v)

#endif // GFT_DEBUG_H
